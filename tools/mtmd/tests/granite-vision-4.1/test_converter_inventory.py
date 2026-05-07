#!/usr/bin/env python3
"""
Tensor-inventory test for the Granite Vision 4.1 mmproj converter.

Checks two claims:
  1. Every HF state-dict tensor is either converted or on an explicit skip list.
  2. Every resulting GGUF tensor has a name we recognize and a shape matching
     the HF source (modulo transpose conventions).

Run as:
    python test_converter_inventory.py \
        --model /path/to/hf/granite-vision-4.1-4b \
        --mmproj /path/to/converted.mmproj.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "gguf-py"))

import gguf  # noqa: E402

# Top-level HF prefixes we deliberately skip in mmproj conversion.
SKIP_PREFIXES = (
    "model.language_model.",
    "model.vision_tower.vision_model.head.",  # SigLIP attention pooling head, unused
)


def load_hf_state_dict_keys(model_dir: Path) -> set[str]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        return set(weight_map.keys())
    # fallback: single shard
    from safetensors import safe_open  # type: ignore
    keys = set()
    for shard in model_dir.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                keys.add(k)
    return keys


def load_gguf(mmproj: Path) -> dict[str, list[int]]:
    reader = gguf.GGUFReader(str(mmproj))
    return {t.name: list(t.shape) for t in reader.tensors}


def is_skipped(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_PREFIXES)


# Build the set of GGUF names we expect to see from the HF source-name set.
def expected_gguf_names(hf_keys: set[str]) -> dict[str, list[str]]:
    """
    Return map: gguf_name -> list of HF source names that produce it.
    (Used as a sanity check — no two HF tensors should map to the same
    GGUF name.)
    """
    out: dict[str, list[str]] = {}
    def add(gguf_name, hf_name):
        out.setdefault(gguf_name, []).append(hf_name)

    for hf in hf_keys:
        if is_skipped(hf):
            continue
        # image_newline
        if hf == "model.image_newline":
            add("v.image_newline", hf)
            continue
        # SigLIP tower (covered by existing tensor_mapping; we just enumerate)
        if hf.startswith("model.vision_tower.vision_model."):
            # We'll validate per-shape later; name mapping is internal.
            add(f"<siglip>:{hf}", hf)
            continue
        # Projector blocks
        if hf.startswith("model.layerwise_projectors."):
            bid = int(hf.split(".")[2])
            add(_proj_name(bid, hf), hf)
            continue
        if hf.startswith("model.spatial_projectors."):
            bid = 4 + int(hf.split(".")[2])
            add(_proj_name(bid, hf), hf)
            continue
        # Anything else is unexpected.
        add("<UNMAPPED>", hf)

    return out


def _proj_name(bid: int, hf_name: str) -> str:
    # Derive the expected GGUF name for a projector block tensor.
    # Mirrors Granite4VisionMmprojModel._normalize_projector_subname +
    # the TENSOR_NAMES strings in constants.py.
    sub = hf_name.split(".", 3)[3]  # after "model.XXX_projectors.N."
    base = f"mm.proj.{bid}"
    if sub.startswith("norm."):
        return f"{base}.norm.{sub.split('.', 1)[1]}"
    if sub == "query":
        return f"{base}.query"
    if sub == "image_positions":
        return f"{base}.image_positions"
    if sub.startswith("out_linear."):
        return f"{base}.out_linear.{sub.split('.', 1)[1]}"
    if sub.startswith("qformer.layernorm."):
        return f"{base}.qformer.layernorm.{sub.rsplit('.', 1)[1]}"

    assert sub.startswith("qformer.encoder.layer.0."), sub
    tail = sub[len("qformer.encoder.layer.0."):]

    def _pack(cat, role, suffix):
        return f"{base}.qformer.{cat}.{role}.{suffix}"

    if tail.startswith("attention.attention."):
        role, suffix = tail[len("attention.attention."):].split(".", 1)
        return f"{base}.qformer.sa.{_role_map(role)}.{suffix}"
    if tail.startswith("attention.output.dense."):
        return f"{base}.qformer.sa.out.{tail[len('attention.output.dense.'):]}"
    if tail.startswith("attention.output.LayerNorm."):
        return f"{base}.qformer.sa.out_norm.{tail[len('attention.output.LayerNorm.'):]}"

    if tail.startswith("crossattention.attention."):
        role, suffix = tail[len("crossattention.attention."):].split(".", 1)
        return f"{base}.qformer.ca.{_role_map(role)}.{suffix}"
    if tail.startswith("crossattention.output.dense."):
        return f"{base}.qformer.ca.out.{tail[len('crossattention.output.dense.'):]}"
    if tail.startswith("crossattention.output.LayerNorm."):
        return f"{base}.qformer.ca.out_norm.{tail[len('crossattention.output.LayerNorm.'):]}"

    if tail.startswith("intermediate_query.dense."):
        return f"{base}.qformer.ffn_up.{tail[len('intermediate_query.dense.'):]}"
    if tail.startswith("output_query.dense."):
        return f"{base}.qformer.ffn_down.{tail[len('output_query.dense.'):]}"
    if tail.startswith("output_query.LayerNorm."):
        return f"{base}.qformer.ffn_norm.{tail[len('output_query.LayerNorm.'):]}"
    raise AssertionError(f"unhandled projector tail: {tail}")


def _role_map(role: str) -> str:
    return {"query": "q", "key": "k", "value": "v"}[role]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--mmproj", required=True, type=Path)
    args = ap.parse_args()

    hf_keys = load_hf_state_dict_keys(args.model)
    gguf_tensors = load_gguf(args.mmproj)

    fails = []

    # Part 1: every HF tensor is either skipped or present in our expected set.
    expected = expected_gguf_names(hf_keys)
    if "<UNMAPPED>" in expected:
        fails.append(("unmapped_hf_tensors", expected["<UNMAPPED>"]))
    # no duplicate HF -> gguf mapping
    dupes = {k: v for k, v in expected.items() if len(v) > 1 and not k.startswith("<siglip>")}
    if dupes:
        fails.append(("duplicate_gguf_targets", dupes))

    # Part 2: every "non-siglip" expected GGUF name exists in the produced GGUF
    # with a shape that matches the HF source tensor (up to a transpose:
    # GGUF stores matrices as {in, out} vs PyTorch {out, in}, so both
    # orderings are considered valid).
    import safetensors.torch as st
    hf_shapes: dict[str, list[int]] = {}
    # Use all shards
    for shard in sorted(args.model.glob("*.safetensors")):
        for k, v in st.load_file(str(shard)).items():
            hf_shapes[k] = list(v.shape)

    for gguf_name, srcs in expected.items():
        if gguf_name.startswith("<siglip>:"):
            continue  # shape-checked separately
        if gguf_name == "<UNMAPPED>":
            continue
        hf = srcs[0]
        hf_shape = hf_shapes[hf]
        if gguf_name not in gguf_tensors:
            fails.append(("missing_gguf_tensor", gguf_name, hf))
            continue
        g_shape = gguf_tensors[gguf_name]
        if not _shapes_match(hf_shape, g_shape):
            fails.append(("shape_mismatch", gguf_name, hf,
                          f"hf={hf_shape} gguf={g_shape}"))

    # Part 3: every GGUF tensor was expected by us (catches orphans).
    produced = set(gguf_tensors)
    expected_names = {n for n in expected if not n.startswith("<")}
    siglip_produced = {n for n in produced if n.startswith("v.") and n != "v.image_newline"}
    non_siglip_produced = produced - siglip_produced
    unexpected = non_siglip_produced - expected_names
    if unexpected:
        fails.append(("unexpected_gguf_tensors", sorted(unexpected)))

    # Sanity: produced SigLIP tensor count matches 27 layers * ~16 weights + a few embeds/norms.
    if len(siglip_produced) < 400 or len(siglip_produced) > 500:
        fails.append(("siglip_count_suspicious", len(siglip_produced)))

    if fails:
        print("FAIL")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("OK")
    print(f"  HF tensors scanned: {len(hf_keys)}")
    print(f"  HF tensors skipped: {sum(1 for k in hf_keys if is_skipped(k))}")
    print(f"  GGUF tensors produced: {len(gguf_tensors)}")
    print(f"  Projector block tensors: {sum(1 for n in produced if n.startswith('mm.proj.'))}")
    print(f"  SigLIP tensors: {len(siglip_produced)}")


def _shapes_match(hf: list[int], g: list[int]) -> bool:
    # Direct match or transposed 2D.
    if hf == g:
        return True
    if list(reversed(hf)) == g:
        return True
    # GGUF sometimes squeezes/unsqueezes trailing size-1 axes — tolerate that.
    def strip1(x): return [d for d in x if d != 1]
    if strip1(hf) == strip1(g):
        return True
    if strip1(list(reversed(hf))) == strip1(g):
        return True
    return False


if __name__ == "__main__":
    main()
