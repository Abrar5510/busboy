"""Export ACT to ONNX -> OpenVINO IR (f32, with end-to-end parity check) and an INT8 IR (NNCF post-training
quantization calibrated on real env observations); export the full SmolVLA sampler to OpenVINO IR (f32 + INT8
weights), parity-checked the same way.

    python -m scripts.export_openvino --act-ckpt outputs/train/act_dinner/checkpoints/last/pretrained_model
    python -m scripts.export_openvino --smolvla-ckpt outputs/train/smolvla_dinner/checkpoints/last/pretrained_model

Writes results/act_dinner*.xml, results/smolvla_dinner*.xml (+ .bin) and results/openvino_export.json.
Runs anywhere (no Intel hardware needed); the INT8 closed-loop check is `scripts.eval --ir results/act_dinner_int8.xml`.
"""

import argparse
import json
import traceback
from pathlib import Path

import openvino as ov
import torch
from lerobot.policies import get_policy_class, make_pre_post_processors

from dinner.env import TASKS, DinnerEnv
from scripts.eval import to_batch

PARITY_TOL = 1e-3
CALIB_SEEDS = 300  # NNCF calibration samples (distinct randomized scenes, mid-episode arm poses)


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
        return pre(to_batch(env.reset(seed, "heavy", "set_table"), text))

    def ir_inputs(b):
        return [b["observation.state"].numpy()] + [b[k].numpy() for k in wrapper.keys]

    b = processed(20000)
    example = (b["observation.state"], *[b[k] for k in wrapper.keys])
    names = ["state"] + [f"img_{k.split('.')[-1]}" for k in wrapper.keys]
    onnx_path = out_xml.with_suffix(".onnx")
    with torch.inference_mode():
        torch.onnx.export(wrapper, example, str(onnx_path), input_names=names, output_names=["action"],
                          opset_version=17, dynamo=False)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(out_xml))
    # Pin f32: OpenVINO's CPU default is f16 on ARM (and bf16 on AMX Xeons), which alone breaks 1e-3 parity.
    core = ov.Core()
    compiled = core.compile_model(str(out_xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})

    # End-to-end parity: full torch pipeline vs processors + IR, first action of the chunk.
    errs = []
    for s in range(parity_n):
        b = processed(20001 + s)
        policy.reset()
        with torch.inference_mode():
            a_torch = post(policy.select_action(dict(b)))
        a_ir = post(torch.from_numpy(compiled(ir_inputs(b))[0][:, 0]))
        errs.append(float((a_torch - a_ir).abs().max()))
    ok = max(errs) < PARITY_TOL
    print(f"ACT parity max abs err {max(errs):.2e} ({'PASS' if ok else 'FAIL'}; suspect double/missing normalization first)")
    report = {"ir": str(out_xml), "inputs": names, "parity_max_abs_err": max(errs), "parity_tol": PARITY_TOL,
              "parity_pass": ok}

    # INT8: NNCF post-training quantization, calibrated on processed observations from randomized scenes.
    try:
        import nncf

        int8_xml = out_xml.with_name(out_xml.stem + "_int8.xml")
        try:  # full INT8 (weights + activations), transformer-aware placement of quantizers
            calib = [ir_inputs(processed(30000 + s)) for s in range(CALIB_SEEDS)]
            int8 = nncf.quantize(core.read_model(str(out_xml)), nncf.Dataset(calib),
                                 model_type=nncf.ModelType.TRANSFORMER, subset_size=len(calib))
            ov.save_model(int8, str(int8_xml))
            c8 = core.compile_model(str(int8_xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            mode = f"weights+activations INT8 (NNCF PTQ, transformer mode, {len(calib)} calibration scenes)"
        except Exception as e:
            # Default-mode PTQ produced a graph the ARM CPU plugin could not compile; weight-only INT8 compiles
            # everywhere and halves the model size.
            print(f"full INT8 unavailable ({str(e).splitlines()[-1][:120]}); falling back to weight-only INT8")
            int8 = nncf.compress_weights(core.read_model(str(out_xml)), mode=nncf.CompressWeightsMode.INT8_ASYM)
            ov.save_model(int8, str(int8_xml))
            c8 = core.compile_model(str(int8_xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            mode = "weight-only INT8 (NNCF compress_weights, int8_asym per-channel)"
        errs8 = []
        for s in range(parity_n):
            b = processed(20001 + s)
            a32 = post(torch.from_numpy(compiled(ir_inputs(b))[0][:, 0]))
            a8 = post(torch.from_numpy(c8(ir_inputs(b))[0][:, 0]))
            errs8.append(float((a32 - a8).abs().max()))
        report["int8"] = {"ir": str(int8_xml), "mode": mode,
                          "max_abs_action_err_vs_f32_rad": max(errs8),
                          "size_mb": {"f32": out_xml.with_suffix(".bin").stat().st_size / 1e6,
                                      "int8": int8_xml.with_suffix(".bin").stat().st_size / 1e6}}
        print(f"INT8 IR written; max action err vs f32 {max(errs8):.3f} rad (task quality is judged closed-loop)")
    except Exception:
        report["int8"] = {"error": traceback.format_exc()[-1500:]}
    return report


class SmolVLAWrapper(torch.nn.Module):
    """Positional (state, lang_tokens, lang_mask, noise, *images in config.image_features order) -> action chunk
    (normalized space). The whole sampler is one graph: SigLIP + SmolVLM prefix (KV cache built once), then
    config.num_steps Euler steps of the action expert, unrolled at trace time. Noise is an input so the IR is
    deterministic and parity-checkable; at run time the caller draws it."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.keys = list(policy.config.image_features)

    def forward(self, state, lang_tokens, lang_mask, noise, *images):
        p = self.policy
        batch = {"observation.state": state, **dict(zip(self.keys, images))}
        imgs, img_masks = p.prepare_images(batch)
        actions = p.model.sample_actions(imgs, img_masks, lang_tokens, lang_mask, p.prepare_state(batch), noise=noise)
        return actions[:, :, :p.config.action_feature.shape[0]]


def smolvla_inputs(policy, b, noise):
    keys = list(policy.config.image_features)
    return ([b["observation.state"], b["observation.language.tokens"], b["observation.language.attention_mask"].bool(),
             noise] + [b[k] for k in keys])


def export_smolvla(ckpt, out_dir, parity_n):
    """Full SmolVLA -> OpenVINO IR (f32, parity-checked) + INT8 weight-compressed IR."""
    # float(): the checkpoint loads the VLM in bf16; trace (and compare) in f32 so parity measures the export,
    # not bf16 rounding (bf16 torch vs f32 IR differs by ~3e-3 rad after 10 flow steps).
    policy = get_policy_class("smolvla").from_pretrained(ckpt).to("cpu").float().eval()
    pre, post = make_pre_post_processors(policy.config, ckpt,
                                      preprocessor_overrides={"device_processor": {"device": "cpu"}})
    wrapper = SmolVLAWrapper(policy).eval()
    env = DinnerEnv()
    cfg = policy.config
    shape = (1, cfg.chunk_size, cfg.max_action_dim)

    def processed(seed, task="set_table"):
        return pre(to_batch(env.reset(seed, "nominal", task), TASKS[task]["train"][0]))

    # transformers' mask helper breaks under tracing (a traced length is a 0-d tensor). Fixed-size images make
    # every vision patch valid, so the SigLIP mask is all-ones and dropping it is exact (the parity check agrees).
    import transformers.models.smolvlm.modeling_smolvlm as smolvlm

    smolvlm.create_bidirectional_mask = lambda **kw: None
    # The KV cache starts as an empty 1-D tensor and concatenates onto it, which OpenVINO's Concat rejects;
    # take the first update as-is instead (same values).
    from transformers.cache_utils import DynamicLayer

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized or self.keys.numel() == 0:
            self.lazy_initialization(key_states, value_states)
            self.keys, self.values = key_states, value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    DynamicLayer.update = update

    torch.manual_seed(0)
    example = smolvla_inputs(policy, processed(20000), torch.randn(shape))
    xml = out_dir / "smolvla_dinner.xml"
    names = ["state", "lang_tokens", "lang_mask", "noise"] + [f"img_{k.split('.')[-1]}" for k in wrapper.keys]
    with torch.inference_mode():  # static shapes: the trace bakes sequence lengths into the attention masks
        ov_model = ov.convert_model(wrapper, example_input=tuple(example),
                                    input=[ov.PartialShape(list(t.shape)) for t in example])
    for inp, name in zip(ov_model.inputs, names):
        inp.get_tensor().set_names({name})
    ov.save_model(ov_model, str(xml))  # weights saved as f16 by default (compress_to_fp16); run in f32
    core = ov.Core()
    compiled = core.compile_model(str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})

    def ir(c, inputs):
        return torch.from_numpy(c([t.numpy() for t in inputs])[0])

    errs, errs_rad = [], []  # whole chunk in normalized space; first action in radians (what the robot gets)
    for s in range(parity_n):
        noise = torch.randn(shape)
        inputs = smolvla_inputs(policy, processed(20001 + s, list(TASKS)[s % len(TASKS)]), noise)
        with torch.inference_mode():
            ref = wrapper(*inputs)
        out = ir(compiled, inputs)
        errs.append(float((ref - out).abs().max()))
        errs_rad.append(float((post(ref[:, 0]) - post(out[:, 0])).abs().max()))
    report = {"ir": str(xml), "inputs": names, "num_steps": cfg.num_steps, "chunk_size": cfg.chunk_size,
              "parity_max_abs_err": max(errs_rad), "parity_max_abs_err_normalized_chunk": max(errs),
              "parity_tol": PARITY_TOL, "parity_pass": max(errs_rad) < PARITY_TOL, "reference": "PyTorch f32"}
    print(f"SmolVLA parity: first action {max(errs_rad):.2e} rad, whole chunk {max(errs):.2e} normalized")

    import nncf

    int8_xml = out_dir / "smolvla_dinner_int8.xml"
    ov.save_model(nncf.compress_weights(core.read_model(str(xml)), mode=nncf.CompressWeightsMode.INT8_ASYM),
                  str(int8_xml))
    c8 = core.compile_model(str(int8_xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    errs8 = []
    for s in range(parity_n):
        noise = torch.randn(shape)
        inputs = smolvla_inputs(policy, processed(20001 + s), noise)
        errs8.append(float((post(ir(compiled, inputs)[:, 0]) - post(ir(c8, inputs)[:, 0])).abs().max()))
    report["int8"] = {"ir": str(int8_xml), "mode": "weight-only INT8 (NNCF compress_weights, int8_asym)",
                      "max_abs_action_err_vs_f32_rad": max(errs8),
                      "size_mb": {"ir_fp16_weights": xml.with_suffix(".bin").stat().st_size / 1e6,
                                  "int8": int8_xml.with_suffix(".bin").stat().st_size / 1e6}}
    print(f"SmolVLA INT8 written; first-action err vs f32 {max(errs8):.3f} rad")
    return report


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
        report["smolvla"] = export_smolvla(args.smolvla_ckpt, out, args.parity_n)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
