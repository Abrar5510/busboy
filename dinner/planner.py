"""Language planner: a small instruction-tuned LLM on OpenVINO GenAI turns a free-form command plus the perceived
scene into pick-and-place steps, constrained to a JSON schema so every output parses. A deterministic scheduler
then decides what runs in parallel.

    python -m dinner.planner "Hand the fork from arm B to arm A, then put it left of the plate."
    python -m dinner.planner --check          # plan accuracy over every task paraphrase + extra commands

Split of responsibilities (small models are reliable at the first, not the second):
- LLM: fills a fixed form per object (move? which arm? which target? hand-off?) plus the order.
- critique: checks the plan against the command's words (objects named, hand-off verbs) and, on a mismatch,
  asks the LLM once more with that feedback. It never edits the plan.
- form_to_steps: expands hand-offs into giver -> pad, receiver -> target.
- Scheduler (steps_to_phases): adjacent steps on different arms and different objects run at the same time;
  a hand-off pad step never shares a phase, so the receiving arm always waits for the giver.

Model: OpenVINO/Qwen3-1.7B-int4-ov (INT4 weights, non-thinking mode), downloaded by scripts/get_models.sh into
models/. Plan accuracy on the 29 --check commands: Qwen3-1.7B 24/29, Qwen2.5-1.5B-Instruct 18-22/29 (prompt
variants). DINNER_PLANNER_MODEL selects another model directory (e.g. Qwen3-4B on an Intel machine).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.environ.get("DINNER_PLANNER_MODEL",
                               Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-int4-ov"))
TARGETS = {"left_of_plate": "fork_target", "right_of_plate": "spoon_target",
           "top_right_of_plate": "cup_target", "handoff_pad": "relay"}
CACHE = Path(__file__).resolve().parent.parent / "outputs" / "planner_cache.json"

PLACES = {"left_of_plate": "fork_target", "right_of_plate": "spoon_target", "top_right_of_plate": "cup_target"}
ITEMS = ("fork", "spoon", "cup")
# A fixed form per object instead of a free list: small models fill slots far more reliably than they write
# ordered step lists (no invented extra steps, no half-written hand-offs).
_SLOT = {"type": "object",
         "properties": {"move": {"type": "boolean"}, "hand_off": {"type": "boolean"},
                        "arm": {"enum": ["left", "right"]}, "target": {"enum": list(PLACES)}},
         "required": ["move", "hand_off", "arm", "target"]}
SCHEMA = {"type": "object",
          "properties": {**{o: _SLOT for o in ITEMS},
                         "order": {"type": "array", "items": {"enum": list(ITEMS)}, "maxItems": 3}},
          "required": [*ITEMS, "order"]}

SYSTEM = """You control two robot arms setting a dinner table. Arm A is the left arm, arm B is the right arm.
Objects: fork, spoon, cup. Fill in one entry per object:
- move: true only if the command asks to move this object. "Set the table", "cutlery" and "silverware" mean ONLY the fork and the spoon, so the cup gets move=false unless the word "cup" is in the command. Objects the scene says are already in place: move=false.
- hand_off: true if the command says hand, pass, give, transfer, or "both arms" for this object: one arm picks it up and passes it to the other arm.
- arm: the arm that puts the object down at its target (with a hand-off: the arm that RECEIVES it). Default: fork left, spoon right, cup right. If the command names an arm for this object, use it. "Hand the fork from arm B to arm A" means hand_off=true, arm=left.
- target: default fork left_of_plate, spoon right_of_plate, cup top_right_of_plate, unless the command names another place.
- order: the objects to move, in the order the command gives them.

Examples:
Command: Set the table.
{"fork":{"move":true,"hand_off":false,"arm":"left","target":"left_of_plate"},"spoon":{"move":true,"hand_off":false,"arm":"right","target":"right_of_plate"},"cup":{"move":false,"hand_off":false,"arm":"right","target":"top_right_of_plate"},"order":["fork","spoon"]}
Command: Pass the spoon to arm A and put it left of the plate.
{"fork":{"move":false,"hand_off":false,"arm":"left","target":"left_of_plate"},"spoon":{"move":true,"hand_off":true,"arm":"left","target":"left_of_plate"},"cup":{"move":false,"hand_off":false,"arm":"right","target":"top_right_of_plate"},"order":["spoon"]}
Command: Put the cup at the top right, then the spoon to the right of the plate.
{"fork":{"move":false,"hand_off":false,"arm":"left","target":"left_of_plate"},"spoon":{"move":true,"hand_off":false,"arm":"right","target":"right_of_plate"},"cup":{"move":true,"hand_off":false,"arm":"right","target":"top_right_of_plate"},"order":["cup","spoon"]}"""


def form_to_steps(form):
    """Filled form -> ordered [{arm, object, target}] with hand-offs expanded over the relay pad."""
    order = [o for o in form["order"] if form[o]["move"]]
    order = list(dict.fromkeys(order + [o for o in ITEMS if form[o]["move"]]))  # dedupe; append forgotten ones
    steps = []
    for o in order:
        f = form[o]
        if f["hand_off"]:
            steps.append({"arm": "right" if f["arm"] == "left" else "left", "object": o, "target": "handoff_pad"})
        steps.append({"arm": f["arm"], "object": o, "target": f["target"]})
    return steps


def describe_scene(scene, done=()):
    """Perceived scene -> short text for the prompt. `done`: objects already verified at a target."""
    lines = []
    for name in ("fork", "spoon", "cup"):
        o = scene["objects"].get(name)
        if o is None:
            lines.append(f"- {name}: not visible")
            continue
        side = "left" if o["pos"][0] < 0 else "right"
        state = "already in place" if name in done else f"on the {side} side of the table"
        extra = "" if o.get("upright", True) else ", tipped over"
        lines.append(f"- {name}: {state}{extra}")
    return "Scene (from the overhead camera):\n" + "\n".join(lines)


def steps_to_phases(steps):
    """[{arm, object, target}] -> [{arm: (object, site)}]; see the module docstring for the parallel rule."""
    phases = []
    for s in steps:
        step = (s["object"], TARGETS[s["target"]])
        prev = phases[-1] if phases else None
        if (prev and s["arm"] not in prev and all(o != step[0] for o, _ in prev.values())
                and step[1] != "relay" and all(site != "relay" for _, site in prev.values())):
            prev[s["arm"]] = step
        else:
            phases.append({s["arm"]: step})
    return phases


class Planner:
    def __init__(self, model_dir=MODEL_DIR, device="CPU", use_cache=True, critic=True):
        import openvino_genai as og

        self.model_dir, self.use_cache, self.critic = Path(model_dir), use_cache, critic
        # Device-default precision. On an ARM CPU (the dev Mac) OpenVINO has no INT4 kernels and unpacks the weights
        # (Qwen3-1.7B: ~9 GB RAM; 4B does not fit in 16 GB), and Qwen2.5 needs DINNER_PLANNER_PRECISION=f32 there
        # or its f16 output is garbage. Intel CPUs/GPUs/NPUs run the INT4 weights directly.
        prec = os.environ.get("DINNER_PLANNER_PRECISION")
        cfg = {"INFERENCE_PRECISION_HINT": prec} if prec else {}
        self.device = device
        self._load = lambda: og.LLMPipeline(str(model_dir), device, **cfg)
        self.pipe = None if use_cache else self._load()  # with the cache, load only on the first miss
        self.gen = og.GenerationConfig(max_new_tokens=200, do_sample=False)
        self.gen.structured_output_config = og.StructuredOutputConfig(json_schema=json.dumps(SCHEMA))
        self.tok = og.Tokenizer(str(model_dir))
        self.latency_s = []

    def _generate(self, messages):
        self.pipe = self.pipe or self._load()
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                              extra_context={"enable_thinking": False})  # Qwen3: answer directly
        tic = time.perf_counter()
        out = str(self.pipe.generate(prompt, self.gen))
        self.latency_s.append(time.perf_counter() - tic)
        return out

    def plan(self, command, scene=None, done=()):
        """-> (steps, phases, notes). One critic round (see critique) re-asks the LLM with feedback.
        Greedy decoding is deterministic, so results are cached by (model, prompt); `latency_s` only records real
        generations. With critic=False (self.critic) the raw first answer is used."""
        user = (describe_scene(scene, done) + "\n" if scene is not None else "") + f"Command: {command}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        key = hashlib.sha256(f"{self.model_dir.name}\n{SYSTEM}\n{user}".encode()).hexdigest()
        cache = json.loads(CACHE.read_text()) if self.use_cache and CACHE.exists() else {}
        entry = cache.get(key)
        if entry is None:
            first = self._generate(messages)
            entry = {"model": self.model_dir.name, "user": user, "output": first, "device": self.device,
                     "latency_s": self.latency_s[-1]}
            issues = critique(command, form_to_steps(json.loads(first)), done)
            if issues:
                entry["critic"] = issues
                entry["revised"] = self._generate(messages + [
                    {"role": "assistant", "content": first},
                    {"role": "user", "content": "Check your answer against the command: " + " ".join(issues)
                                                + " Answer again with the corrected JSON."}])
            if self.use_cache:
                cache[key] = entry
                CACHE.parent.mkdir(exist_ok=True)
                CACHE.write_text(json.dumps(cache, indent=1))
        out = entry.get("revised", entry["output"]) if self.critic else entry["output"]
        steps = form_to_steps(json.loads(out))
        notes = [f"critic: {i}" for i in entry.get("critic", [])] if self.critic else []
        return steps, steps_to_phases(steps), notes


HANDOFF_WORDS = ("hand", "pass", "transfer", "give", "both arms")


def critique(command, steps, done=()):
    """Cheap consistency checks between the command's words and a plan; returns feedback sentences for the LLM.
    It never edits the plan itself: the LLM revises (or doesn't)."""
    text = command.lower()
    named = {o for o in ITEMS if o in text}
    if any(w in text for w in ("table", "cutlery", "silverware")):
        named |= {"fork", "spoon"}
    moved = {s["object"] for s in steps}
    issues = [f"The command asks to move the {o}, but your plan does not." for o in sorted(named - moved - set(done))]
    issues += [f"The command does not ask to move the {o}, but your plan does." for o in sorted(moved - named)]
    issues += [f"The {o} is already in place, so do not move it." for o in sorted(moved & set(done))]
    if any(w in text for w in HANDOFF_WORDS) and not any(s["target"] == "handoff_pad" for s in steps):
        issues.append("The command asks for a hand-off between the arms, but no object has hand_off=true.")
    return issues


# Commands with the phases a correct plan must produce. The train/eval paraphrases of dinner.env.TASKS are
# checked too (see __main__); these add arm naming and compositions no task was defined with.
CASES = [
    ("Open with arm A: put the fork left of the plate.", [{"left": ("fork", "fork_target")}]),
    ("Arm B, place the spoon to the right of the plate.", [{"right": ("spoon", "spoon_target")}]),
    ("Hand the fork from arm B to arm A and place it left of the plate.",
     [{"right": ("fork", "relay")}, {"left": ("fork", "fork_target")}]),
    ("With arm B put the cup top right, and then with arm A put the fork on the left of the plate.",
     [{"right": ("cup", "cup_target"), "left": ("fork", "fork_target")}]),  # independent: the scheduler parallelizes
    ("Set the table, then put the cup at the top right.",
     [{"left": ("fork", "fork_target"), "right": ("spoon", "spoon_target")}, {"right": ("cup", "cup_target")}]),
]


if __name__ == "__main__":
    from dinner.env import TASKS, task_phases

    st = lambda arm, obj, tgt: {"arm": arm, "object": obj, "target": tgt}
    assert steps_to_phases([st("left", "fork", "left_of_plate"), st("right", "spoon", "right_of_plate"),
                            st("right", "cup", "top_right_of_plate")]) \
        == [{"left": ("fork", "fork_target"), "right": ("spoon", "spoon_target")}, {"right": ("cup", "cup_target")}]
    slot = lambda move, arm, tgt, ho=False: {"move": move, "arm": arm, "target": tgt, "hand_off": ho}
    assert form_to_steps({"fork": slot(True, "left", "left_of_plate", True), "spoon": slot(False, "right", "right_of_plate"),
                          "cup": slot(True, "right", "top_right_of_plate"), "order": ["cup", "cup"]}) \
        == [st("right", "cup", "top_right_of_plate"), st("right", "fork", "handoff_pad"), st("left", "fork", "left_of_plate")]
    assert critique("Set the table, then the cup.", [st("right", "cup", "top_right_of_plate")]) == [
        "The command asks to move the fork, but your plan does not.",
        "The command asks to move the spoon, but your plan does not."]
    assert critique("Pass the fork to arm A.", [st("left", "fork", "left_of_plate")])[-1].startswith("The command asks for a hand-off")
    assert steps_to_phases([st("right", "fork", "handoff_pad"), st("left", "fork", "left_of_plate")]) \
        == [{"right": ("fork", "relay")}, {"left": ("fork", "fork_target")}]

    planner = Planner(device=sys.argv[2] if len(sys.argv) > 2 else "CPU")  # cached: the check also warms the agent
    if len(sys.argv) > 1 and sys.argv[1] != "--check":
        steps, phases, notes = planner.plan(sys.argv[1])
        print(json.dumps(steps, indent=1), phases, *notes, f"generations: {planner.latency_s}", sep="\n")
        sys.exit()

    from dinner.env import DinnerEnv
    from dinner.perception import Perception

    env = DinnerEnv(render=False)
    per = Perception(env)
    cases = [(None, text, want) for text, want in CASES]
    for task, spec in TASKS.items():
        cases += [(task, text, task_phases(task)) for text in spec["train"] + spec["eval"]]
    ok = {"raw": 0, "critic": 0}
    for task, text, want in cases:
        env.reset(10000, "nominal", task)  # the scene the command is given in (hand-off tasks swap the cutlery)
        scene = per.observe()
        for mode in ok:
            planner.critic = mode == "critic"
            _, got, notes = planner.plan(text, scene)
            ok[mode] += got == want
            print(("OK  " if got == want else "MISS") + f" [{mode}] {text!r}"
                  + ("" if got == want else f"\n     got  {got}\n     want {want}") + "".join(f"\n     {n}" for n in notes))
    print(f"planner {MODEL_DIR.name}: raw {ok['raw']}/{len(cases)}, with critic {ok['critic']}/{len(cases)} exact; "
          f"p50 {np.median(planner.latency_s):.2f}s per generation on {planner.device}")
