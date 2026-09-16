"""Latency and throughput on every OpenVINO device found: ACT (PyTorch CPU vs OpenVINO f32 / INT8 IRs), the full
SmolVLA sampler IR (f32 / INT8 weights), and with --planner the INT4 LLM planner.

Run this on Intel Core Ultra hardware (CPU / Arc iGPU / NPU) for Intel claims; elsewhere it still runs but the
report marks intel=false.

    python -m scripts.bench_intel --ckpt <act pretrained_model>
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

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
    # One action-chunk inference per call; at 30 Hz control with n_action_steps=100 a chunk covers 3.3 s.
    return {"p50_ms": 1000 * float(np.median(t)), "p95_ms": 1000 * float(np.percentile(t, 95)),
            "throughput_inferences_per_s": n / float(np.sum(t)), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ir", default="results/act_dinner.xml")
    ap.add_argument("--ir-int8", default="results/act_dinner_int8.xml")
    ap.add_argument("--smolvla-ir", default="results/smolvla_dinner.xml", help="skipped if missing")
    ap.add_argument("--smolvla-ckpt", help="also time the PyTorch SmolVLA sampler (f32, CPU) as the baseline")
    ap.add_argument("--planner", action="store_true", help="also time the LLM planner (dinner.planner) per device")
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
    cpu_default = core.get_property("CPU", "INFERENCE_PRECISION_HINT").get_type_name()  # f32 x86, f16 ARM, bf16 AMX
    # (name, IR, device, config). CPU f32 matches torch numerically (parity-checked); other rows trade exactness
    # for speed: device-default precision (f16/bf16), INT8 weights+activations, Arc iGPU, NPU.
    configs = [("cpu_f32", args.ir, "CPU", {"INFERENCE_PRECISION_HINT": "f32"}),
               (f"cpu_{cpu_default}", args.ir, "CPU", {})]
    for dev in core.available_devices:
        if dev.startswith(("GPU", "NPU")):
            configs.append((dev.lower().replace(".", ""), args.ir, dev, {}))
    if Path(args.ir_int8).exists():
        configs.append(("cpu_int8", args.ir_int8, "CPU", {}))
        for dev in core.available_devices:
            if dev.startswith(("GPU", "NPU")):
                configs.append((dev.lower().replace(".", "") + "_int8", args.ir_int8, dev, {}))

    cpu = cpu_model()
    report = {
        "machine": {"cpu": cpu, "platform": platform.platform(), "intel": "intel" in cpu.lower(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "openvino": ov.get_version(),
                    "openvino_cpu_default_precision": cpu_default,
                    "openvino_devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}},
        "act_torch_cpu": bench(torch_fn),
    }
    for name, ir, dev, cfg in configs:
        try:
            c = core.compile_model(ir, dev, {"PERFORMANCE_HINT": "LATENCY", **cfg})
            report[f"act_openvino_{name}"] = bench(lambda c=c: c(np_inputs))
        except Exception as e:  # one device failing to compile this graph shouldn't sink the whole report
            report[f"act_openvino_{name}"] = {"error": str(e)[:300]}
    # SmolVLA full sampler (10 flow-matching steps per chunk): f32 and INT8-weight IRs on every device.
    for ir in (Path(args.smolvla_ir), Path(args.smolvla_ir).with_name(Path(args.smolvla_ir).stem + "_int8.xml")):
        if not ir.exists():
            continue
        m = core.read_model(str(ir))
        feeds = {}
        for inp in m.inputs:
            shp, name = inp.get_partial_shape().to_shape(), inp.get_any_name()
            et = inp.get_element_type().get_type_name()
            feeds[name] = (np.random.rand(*shp).astype(np.float32) if et == "f32" else
                           np.ones(shp, dtype=bool) if et == "boolean" else np.ones(shp, dtype=np.int64))
        tag = "smolvla" + ("_int8" if "int8" in ir.stem else "")
        if args.smolvla_ckpt and tag == "smolvla":
            from scripts.export_openvino import SmolVLAWrapper

            sp = get_policy_class("smolvla").from_pretrained(args.smolvla_ckpt).to("cpu").float().eval()
            wrapper = SmolVLAWrapper(sp).eval()
            tin = [torch.from_numpy(v) for v in feeds.values()]  # IR input order == wrapper argument order

            def smol_torch():
                with torch.inference_mode():
                    wrapper(*tin)

            report["smolvla_torch_cpu_f32"] = bench(smol_torch, warmup=2, n=10)
            del sp, wrapper
        runs = [("cpu_f32", "CPU", {"INFERENCE_PRECISION_HINT": "f32"}), (f"cpu_{cpu_default}", "CPU", {})]
        runs += [(d.lower().replace(".", ""), d, {}) for d in core.available_devices if d.startswith(("GPU", "NPU"))]
        for name, dev, cfg in runs:
            try:
                c = core.compile_model(m, dev, {"PERFORMANCE_HINT": "LATENCY", **cfg})
                report[f"{tag}_openvino_{name}"] = bench(lambda c=c: c(feeds), warmup=2, n=10)
            except Exception as e:
                report[f"{tag}_openvino_{name}"] = {"error": str(e)[:300]}

    if args.planner:
        from dinner.planner import MODEL_DIR, Planner

        for dev in [d for d in core.available_devices if d.startswith(("CPU", "GPU", "NPU"))]:
            try:
                pl = Planner(device=dev, use_cache=False)
                for cmd in ("Set the table.", "Hand the fork from arm B to arm A and place it left of the plate."):
                    pl.plan(cmd)
                report[f"planner_{MODEL_DIR.name}_{dev.lower()}"] = {
                    "p50_s_per_generation": float(np.median(pl.latency_s)), "n": len(pl.latency_s),
                    "weights": "INT4"}
            except Exception as e:
                report[f"planner_{MODEL_DIR.name}_{dev.lower()}"] = {"error": str(e)[:300]}

    report["speedup_p50_vs_torch"] = {k.removeprefix("act_openvino_"): report["act_torch_cpu"]["p50_ms"] / v["p50_ms"]
                                      for k, v in report.items() if k.startswith("act_openvino_") and "p50_ms" in v}
    if "smolvla_torch_cpu_f32" in report:
        report["smolvla_speedup_p50_vs_torch"] = {
            k.removeprefix("smolvla_"): report["smolvla_torch_cpu_f32"]["p50_ms"] / v["p50_ms"]
            for k, v in report.items() if "_openvino_" in k and k.startswith("smolvla") and "p50_ms" in v}
    if not report["machine"]["intel"]:
        print(f"WARNING: CPU is {cpu!r}, not Intel; do not report these numbers as Intel results.")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
