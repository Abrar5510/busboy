"""Print the README results table from results/*.json (so the table is never hand-typed).

    python -m scripts.results_table            # markdown rows
    python -m scripts.results_table --latency  # also the latency/precision lines
"""

import argparse
import glob
import json
from pathlib import Path

from dinner.env import TASKS

ORDER = ["smolvla_torch_nominal_train", "smolvla_torch_nominal_eval", "smolvla_torch_heavy_train",
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
    args = ap.parse_args()

    found = {Path(f).stem: json.load(open(f)) for f in glob.glob(f"{args.results}/*.json")}
    tasks = list(TASKS)
    print("| Policy | Backend | Randomization | Instructions | " + " | ".join(tasks) + " |")
    print("|---" * (4 + len(tasks)) + "|")
    print("| Expert (upper bound) | — | nominal | — | " + " | ".join("10/10" for _ in tasks) + " |")
    for tag in ORDER:
        r = found.get(tag)
        if r is None:
            continue
        instr = {"train": "train", "eval": "held-out", "all": "all"}[r["paraphrases"]]
        row = [r["policy"].upper() if r["policy"] == "act" else "SmolVLA", LABEL.get(r["backend"], r["backend"]),
               r["rand"], instr if r["policy"] != "act" else "—"]
        print("| " + " | ".join(row + [cell(r["tasks"][t]) for t in tasks]) + " |")

    if args.latency:
        print()
        for tag in ORDER:
            r = found.get(tag)
            if r:
                print(f"- `{tag}`: p50 {r['p50_infer_ms']:.0f} ms, p95 {r['p95_infer_ms']:.0f} ms per action chunk "
                      f"on {r['device']}")
        bench = found.get("intel_bench")
        if bench:
            m = bench["machine"]
            print(f"- bench machine: {m['cpu']} (intel={m['intel']}), OpenVINO {m['openvino']}")
            for k, v in bench.items():
                if k.startswith("act_") and "p50_ms" in v:
                    print(f"  - {k.removeprefix('act_')}: p50 {v['p50_ms']:.1f} ms, "
                          f"{v['throughput_inferences_per_s']:.1f} inferences/s")
        exp = found.get("openvino_export", {}).get("act")
        if exp:
            i8 = exp.get("int8", {})
            print(f"- export: parity {exp['parity_max_abs_err']:.1e} rad (pass={exp['parity_pass']}); "
                  f"INT8 {i8.get('mode', 'n/a')}, max action err {i8.get('max_abs_action_err_vs_f32_rad', float('nan')):.3f} rad")


if __name__ == "__main__":
    main()
