"""Print the README results table from results/*.json (so the table is never hand-typed).

    python -m scripts.results_table            # markdown rows
    python -m scripts.results_table --latency  # also the latency/precision lines
    python -m scripts.results_table --latency --readme README.md  # refresh the README block in place
"""

import argparse
import glob
import json
from pathlib import Path

from dinner.env import TASKS

AGENT = [("agent_nominal_train", "nominal", "train", ""), ("agent_nominal_eval", "nominal", "held-out", ""),
         ("agent_heavy_train", "heavy", "train", ""), ("agent_nominal_train_disturb", "nominal", "train", "object knocked off mid-task"),
         ("agent_nominal_train_preplaced", "nominal", "train", "one object already placed")]
ORDER = ["smolvla_torch_nominal_train", "smolvla_openvino-cpu_nominal_train", "smolvla_openvino-cpu-int8_nominal_train",
         "smolvla_torch_nominal_eval", "smolvla_torch_heavy_train",
         "act_torch_nominal_train", "act_openvino-cpu_nominal_train", "act_openvino-cpu-int8_nominal_train",
         "act_torch_heavy_train"]
LABEL = {"torch": "torch", "openvino-cpu": "OpenVINO f32", "openvino-cpu-int8": "OpenVINO INT8"}


def cell(entry):
    if entry == "n/a":
        return "n/a"
    placed = entry.get("object_placed") or {}
    extra = ", ".join(f"{o} {n}" for o, n in placed.items()) if len(placed) > 1 else ""
    return f"{entry['successes']}/{entry['episodes']}" + (f" ({extra})" if extra else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--readme", help="rewrite the block between <!-- results:begin/end --> in this file")
    args = ap.parse_args()
    if args.readme:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.readme, readme = None, args.readme
            report(args)
        text = Path(readme).read_text()
        head, rest = text.split("<!-- results:begin -->")
        _, tail = rest.split("<!-- results:end -->")
        Path(readme).write_text(f"{head}<!-- results:begin -->\n{buf.getvalue()}<!-- results:end -->{tail}")
        return
    report(args)


def report(args):
    found = {Path(f).stem: json.load(open(f)) for f in glob.glob(f"{args.results}/*.json")}
    tasks = list(TASKS)
    print("| Policy | Backend | Randomization | Instructions | " + " | ".join(tasks) + " |")
    print("|---" * (4 + len(tasks)) + "|")
    for level in ("nominal", "heavy"):
        r = found.get(f"expert_{level}")
        if r:
            print(f"| Expert (ground-truth poses, upper bound) | — | {level} | — | "
                  + " | ".join(cell(r["tasks"][t]) for t in tasks) + " |")
    for tag, rand, instr, note in AGENT:
        r = found.get(tag)
        if r:
            label = "**Agent** (LLM planner + perception + skills)" + (f", {note}" if note else "")
            print(f"| {label} | OpenVINO INT4 planner | {rand} | {instr} | "
                  + " | ".join(cell(r["tasks"].get(t, "n/a")) for t in tasks) + " |")
    for tag in ORDER:
        r = found.get(tag)
        if r is None:
            continue
        instr = {"train": "train", "eval": "held-out", "all": "all"}[r["paraphrases"]]
        row = [r["policy"].upper() if r["policy"] == "act" else "SmolVLA", LABEL.get(r["backend"], r["backend"]),
               r["rand"], instr if r["policy"] != "act" else "—"]
        print("| " + " | ".join(row + [cell(r["tasks"][t]) for t in tasks]) + " |")

    extra = found.get("agent_nominal_train_extra")
    if extra:
        print()
        print("| Free-form command (agent) | Success |")
        print("|---|---|")
        for k, v in extra["tasks"].items():
            cmd = extra["episodes"][k][0]["command"]
            print(f"| {cmd} | {v['successes']}/{v['episodes']} |")

    if args.latency:
        print()
        for tag in ORDER:
            r = found.get(tag)
            if r:
                print(f"- `{tag}`: p50 {r['p50_infer_ms']:.0f} ms, p95 {r['p95_infer_ms']:.0f} ms per action chunk "
                      f"on {r['device']}")
        for tag, *_ in AGENT[:1]:
            r = found.get(tag)
            if r and r["planner_latency_s_uncached"]:
                lat = r["planner_latency_s_uncached"]
                print(f"- planner `{r['planner_model']}` on {r['planner_device']}: p50 {sorted(lat)[len(lat) // 2]:.1f} s "
                      f"per generation ({len(lat)} uncached)")
        bench = found.get("intel_bench")
        if bench:
            m = bench["machine"]
            print(f"- bench machine: {m['cpu']} (intel={m['intel']}), OpenVINO {m['openvino']}")
            for k, v in bench.items():
                if isinstance(v, dict) and "p50_ms" in v:
                    print(f"  - {k}: p50 {v['p50_ms']:.1f} ms, "
                          f"{v['throughput_inferences_per_s']:.1f} inferences/s")
        sv = found.get("openvino_export", {}).get("smolvla")
        if sv:
            print(f"- SmolVLA export (whole sampler): parity {sv['parity_max_abs_err']:.1e} rad vs {sv['reference']} "
                  f"(pass={sv['parity_pass']}); {sv['int8']['mode']}, max action err "
                  f"{sv['int8']['max_abs_action_err_vs_f32_rad']:.3f} rad, "
                  f"{sv['int8']['size_mb']['ir_fp16_weights']:.0f} -> {sv['int8']['size_mb']['int8']:.0f} MB")
        exp = found.get("openvino_export", {}).get("act")
        if exp:
            i8 = exp.get("int8", {})
            print(f"- export: parity {exp['parity_max_abs_err']:.1e} rad (pass={exp['parity_pass']}); "
                  f"INT8 {i8.get('mode', 'n/a')}, max action err {i8.get('max_abs_action_err_vs_f32_rad', float('nan')):.3f} rad")


if __name__ == "__main__":
    main()
