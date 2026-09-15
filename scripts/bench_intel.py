"""ACT latency: OpenVINO IR on CPU vs PyTorch on CPU, same machine.

Run this on x86 Intel hardware (Intel Developer Cloud, a borrowed PC, or hackathon nodes) for Intel claims;
elsewhere it still runs but the report marks intel=false.

    python scripts/bench_intel.py --ckpt <act pretrained_model> --ir results/act_dinner.xml
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import openvino as ov
import torch
from lerobot.policies import get_policy_class


def cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        return next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), out[:200])
    except FileNotFoundError:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()


def bench(fn, warmup=10, n=100):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        tic = time.perf_counter()
        fn()
        t.append(time.perf_counter() - tic)
    return {"p50_ms": 1000 * float(np.median(t)), "p95_ms": 1000 * float(np.percentile(t, 95)), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ir", default="results/act_dinner.xml")
    ap.add_argument("--out", default="results/intel_bench.json")
    args = ap.parse_args()

    policy = get_policy_class("act").from_pretrained(args.ckpt).to("cpu").eval()
    keys = list(policy.config.image_features)
    state_shape = tuple(policy.config.input_features["observation.state"].shape)
    torch.manual_seed(0)
    batch = {"observation.state": torch.randn(1, *state_shape),
             **{k: torch.rand(1, *policy.config.input_features[k].shape) for k in keys}}
    np_inputs = [batch["observation.state"].numpy()] + [batch[k].numpy() for k in keys]

    def torch_fn():
        with torch.inference_mode():
            policy.predict_action_chunk(batch)

    core = ov.Core()
    default_precision = core.get_property("CPU", "INFERENCE_PRECISION_HINT")  # ov.Type
    # f32 matches torch numerically (parity-checked); the device default (f16 ARM / bf16 AMX) is faster but approximate.
    precisions = {"f32": "f32", default_precision.get_type_name(): default_precision}
    compiled = {name: core.compile_model(args.ir, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": v})
                for name, v in precisions.items()}
    default_precision = default_precision.get_type_name()

    cpu = cpu_model()
    report = {
        "machine": {"cpu": cpu, "platform": platform.platform(), "intel": "intel" in cpu.lower(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "openvino": ov.get_version(),
                    "openvino_default_precision": default_precision},
        "act_torch_cpu": bench(torch_fn),
    }
    for p, c in compiled.items():
        report[f"act_openvino_cpu_{p}"] = bench(lambda c=c: c(np_inputs))
    report["speedup_p50_f32"] = report["act_torch_cpu"]["p50_ms"] / report["act_openvino_cpu_f32"]["p50_ms"]
    if not report["machine"]["intel"]:
        print(f"WARNING: CPU is {cpu!r}, not Intel; do not report these numbers as Intel results.")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
