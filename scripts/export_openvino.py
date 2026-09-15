"""Export ACT to ONNX -> OpenVINO IR (with end-to-end parity check); attempt SmolVLA export.

    PYTHONPATH=. python scripts/export_openvino.py --act-ckpt outputs/train/act_dinner/checkpoints/last/pretrained_model
    PYTHONPATH=. python scripts/export_openvino.py --smolvla-ckpt outputs/train/smolvla_dinner/checkpoints/last/pretrained_model

Writes results/act_dinner.xml (+ .bin) and results/openvino_export.json.
Runs anywhere (no Intel hardware needed); only scripts/bench_intel.py must run on Intel silicon.
"""

import argparse
import json
import shutil
import subprocess
import traceback
from pathlib import Path

import openvino as ov
import torch
from lerobot.policies import get_policy_class, make_pre_post_processors

from dinner.env import TASKS, DinnerEnv
from scripts.eval import to_batch

PARITY_TOL = 1e-3


class ACTWrapper(torch.nn.Module):
    """Positional (state, *images in config.image_features order) -> action chunk (normalized space).

    Wraps predict_action_chunk, not select_action: select_action keeps a Python deque action queue
    that tracing would bake in or break on. In lerobot 0.6.1 ACT normalization lives in the pre/post
    processors (not inside the policy), so the IR takes processor outputs and the processors stay in
    Python around it. The parity check below is the arbiter: normalization must be applied exactly once."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.keys = list(policy.config.image_features)

    def forward(self, state, *images):
        return self.policy.predict_action_chunk({"observation.state": state, **dict(zip(self.keys, images))})


def export_act(ckpt, out_xml, parity_n):
    policy = get_policy_class("act").from_pretrained(ckpt).to("cpu").eval()
    pre, post = make_pre_post_processors(policy.config, ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    print("input_features:", policy.config.input_features)
    wrapper = ACTWrapper(policy).eval()
    env = DinnerEnv()
    text = TASKS["set_table"]["train"][0]

    def processed(seed):
        return pre(to_batch(env.reset(seed, "heavy"), text))

    b = processed(20000)
    example = (b["observation.state"], *[b[k] for k in wrapper.keys])
    names = ["state"] + [f"img_{k.split('.')[-1]}" for k in wrapper.keys]
    onnx_path = out_xml.with_suffix(".onnx")
    with torch.inference_mode():
        torch.onnx.export(wrapper, example, str(onnx_path), input_names=names, output_names=["action"],
                          opset_version=17, dynamo=False)
    ov.save_model(ov.convert_model(str(onnx_path)), str(out_xml))
    compiled = ov.Core().compile_model(str(out_xml), "CPU")

    # End-to-end parity: full torch pipeline vs processors + IR, first action of the chunk.
    errs = []
    for s in range(parity_n):
        b = processed(20001 + s)
        policy.reset()
        with torch.inference_mode():
            a_torch = post(policy.select_action(dict(b)))
        ir = compiled([b["observation.state"].numpy()] + [b[k].numpy() for k in wrapper.keys])[0]
        a_ir = post(torch.from_numpy(ir[:, 0]))
        errs.append(float((a_torch - a_ir).abs().max()))
    ok = max(errs) < PARITY_TOL
    print(f"ACT parity max abs err {max(errs):.2e} ({'PASS' if ok else 'FAIL'}; suspect double/missing normalization first)")
    return {"ir": str(out_xml), "inputs": names, "parity_max_abs_err": max(errs), "parity_tol": PARITY_TOL, "parity_pass": ok}


def export_smolvla(ckpt, out_dir):
    status = {}
    cli = shutil.which("optimum-cli")
    if cli is None:
        status["optimum_cli"] = ("not attempted: optimum-intel not installed. A LeRobot SmolVLA checkpoint has no "
                                 "transformers model_type, so optimum's exporter has no architecture to map it to.")
    else:
        r = subprocess.run([cli, "export", "openvino", "-m", str(ckpt), str(out_dir / "smolvla_ov")],
                           capture_output=True, text=True, timeout=3600)
        status["optimum_cli"] = {"returncode": r.returncode, "stderr_tail": r.stderr[-1500:]}
        if r.returncode == 0:
            return status

    # Fallback: the frozen vision encoder alone (the flow-matching action expert loops over denoising steps).
    try:
        policy = get_policy_class("smolvla").from_pretrained(ckpt).to("cpu").eval()
        name, vision = next((n, m) for n, m in policy.named_modules() if n.endswith("vision_model"))
        example = torch.rand(1, 3, *policy.config.resize_imgs_with_padding)
        xml = out_dir / "smolvla_vision_encoder.xml"
        with torch.inference_mode():
            ov.save_model(ov.convert_model(vision, example_input=example), str(xml))
            ref = vision(example)
        ref = ref.last_hidden_state if hasattr(ref, "last_hidden_state") else ref
        out = ov.Core().compile_model(str(xml), "CPU")([example.numpy()])[0]
        err = float((ref - torch.from_numpy(out)).abs().max())
        status["vision_encoder"] = {"module": name, "ir": str(xml), "parity_max_abs_err": err}
    except Exception:
        status["vision_encoder"] = {"error": traceback.format_exc()[-1500:]}
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-ckpt")
    ap.add_argument("--smolvla-ckpt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--parity-n", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    report_path = out / "openvino_export.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["openvino_version"] = ov.get_version()
    if args.act_ckpt:
        report["act"] = export_act(args.act_ckpt, out / "act_dinner.xml", args.parity_n)
    if args.smolvla_ckpt:
        report["smolvla"] = export_smolvla(args.smolvla_ckpt, out)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
