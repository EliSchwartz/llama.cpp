#!/usr/bin/env python3
"""
Generate input/output fixtures from the reference HF Granite Vision 4.1 implementation.

Each fixture captures (input_args, output_tensor) pairs at well-defined points
in the HF forward pass so that llama.cpp ports of each component can be
numerically validated against the reference.

Usage:
    python gen_fixtures.py --model-dir /path/to/granite-vision-4.1-4b --out ./fixtures

Fixtures produced:
    fixtures/
      meta.json                       (config snapshot, component shape metadata)
      siglip_input.npy                (B, C, H, W) pixel values (B=1 tile, 384x384)
      siglip_hidden_states.npy        (num_layers+1, B, T, D_vision) all SigLIP hidden states
      downsampler_interp_<i>_in.npy   layer-i layerwise projector input (B, T, D_vision)
      downsampler_interp_<i>_out.npy  layer-i layerwise projector output (B, T_q, D_llm)
      downsampler_spatial_<i>_in.npy  spatial projector i input
      downsampler_spatial_<i>_out.npy spatial projector i output
      pack_<k>_features_in.npy        input for pack_and_unpad call k
      pack_<k>_image_sizes.npy        image_sizes argument
      pack_<k>_out.npy                packed output
      image_newline.npy               the learned image_newline vector
      pinpoints_<WxH>.json            for each test size, expected num_patches
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


def load_model(model_dir: str):
    """Load the HF model in float32 on CPU for reproducibility."""
    sys.path.insert(0, model_dir)
    from transformers import AutoModel, AutoProcessor, AutoConfig  # noqa: E402

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor, config


def make_test_image(w: int = 384, h: int = 384) -> Image.Image:
    """Deterministic test image."""
    deterministic()
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_npy(path: Path, t: torch.Tensor | np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    np.save(path, t)


def dump_siglip(model, out_dir: Path):
    """Run SigLIP vision tower on a canonical 384x384 image and dump every hidden state."""
    deterministic()
    # Pixel values: batch of 1, channels 3, 384x384, normalized to [-1, 1] with mean/std 0.5.
    pixel_values = torch.randn(1, 3, 384, 384)
    save_npy(out_dir / "siglip_input.npy", pixel_values)

    vt = model.model.vision_tower
    with torch.no_grad():
        out = vt(pixel_values, output_hidden_states=True)
    hs = torch.stack(list(out.hidden_states), dim=0)  # (L+1, B, T, D)
    save_npy(out_dir / "siglip_hidden_states.npy", hs)
    return hs, pixel_values


def dump_downsamplers(model, hidden_states: torch.Tensor, out_dir: Path, cfg):
    """
    For each layerwise and spatial projector, dump its (input, output) pair.

    hidden_states: (L+1, B, T, D_vision) from the SigLIP encoder.
    """
    # Layerwise projectors: one per (vision_layer, llm_layer) pair in deepstack_layer_map.
    for i, (vision_layer, _llm_layer) in enumerate(cfg.deepstack_layer_map):
        x = hidden_states[vision_layer]  # (B, T, D)
        # default feature-select-strategy removes CLS; SigLIP has no CLS so "full" and "default"
        # produce the same T, but follow the config.
        if cfg.vision_feature_select_strategy == "default":
            x = x[:, 1:]
        save_npy(out_dir / f"downsampler_interp_{i}_in.npy", x)
        with torch.no_grad():
            y = model.model.layerwise_projectors[i](x)
        save_npy(out_dir / f"downsampler_interp_{i}_out.npy", y)

    # Spatial projectors: 4 offsets.
    if cfg.use_spatial_sampling:
        x = hidden_states[cfg.spatial_vision_layer]
        if cfg.vision_feature_select_strategy == "default":
            x = x[:, 1:]
        for i in range(4):
            save_npy(out_dir / f"downsampler_spatial_{i}_in.npy", x)
            with torch.no_grad():
                y = model.model.spatial_projectors[i](x)
            save_npy(out_dir / f"downsampler_spatial_{i}_out.npy", y)


def dump_downsampler_substeps(model, hidden_states: torch.Tensor, out_dir: Path, cfg):
    """
    Dump the intermediate tensors of a single interp and a single spatial
    WindowQFormerDownsampler block (block indices 0 and 0 respectively) so
    the C++ port can validate each sub-step independently instead of
    bisecting a >20-op forward from in/out alone.

    For each block we save (all as float32 .npy, B=1, no dropout):
      <prefix>_norm.npy          LayerNorm(hidden_states)
      <prefix>_enc.npy           _win(normed, 24, 8)   -> (9, 64, 1152)
      <prefix>_downsampled.npy   downsampler(normed)   -> (1, 144, 1152)
      <prefix>_query_embeds.npy  query + _win(downsampled, 12, 4)  -> (9, 16, 1152)
      <prefix>_encoder_embeds.npy enc + image_positions            -> (9, 64, 1152)
      <prefix>_qformer_out.npy   qformer(q=query, enc=encoder).last_hidden_state -> (9, 16, 1152)
      <prefix>_unwin.npy         _unwin(qformer_out)   -> (1, 144, 1152)
      <prefix>_out.npy           out_linear(unwin)     -> (1, 144, 2560)

    The _out file is redundant with downsampler_{interp,spatial}_0_out.npy
    but is produced here from the captured intermediates to confirm the
    chain matches end-to-end.
    """
    def run_one(block, x, prefix):
        # Recreate the forward step-by-step while stashing intermediates.
        x = x.clone()
        # Top-level norm
        normed = block.norm(x)
        save_npy(out_dir / f"{prefix}_norm.npy", normed)

        side = block.image_side
        win  = block.window_side
        qside = block.query_side
        n = side // win
        new_side = n * qside

        enc = block._win(normed, side, win)
        save_npy(out_dir / f"{prefix}_enc.npy", enc)

        downsampled = block.downsampler(normed)
        save_npy(out_dir / f"{prefix}_downsampled.npy", downsampled)

        downsampled_w = block._win(downsampled, new_side, qside)
        query_embeds = block.query + downsampled_w
        save_npy(out_dir / f"{prefix}_query_embeds.npy", query_embeds)

        encoder_embeds = enc + block.image_positions
        # Skip dropout (block.dropout) because eval() is set but we still
        # want the exact sum to land in the fixture.
        save_npy(out_dir / f"{prefix}_encoder_embeds.npy", encoder_embeds)

        out_w = block.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_embeds,
            return_dict=True,
        ).last_hidden_state
        save_npy(out_dir / f"{prefix}_qformer_out.npy", out_w)

        unwinned = block._unwin(out_w, n=n, win=qside)
        save_npy(out_dir / f"{prefix}_unwin.npy", unwinned)

        final = block.out_linear(unwinned)
        save_npy(out_dir / f"{prefix}_out.npy", final)

    # All interp blocks
    for i, (vlayer, _) in enumerate(cfg.deepstack_layer_map):
        interp_block = model.model.layerwise_projectors[i]
        x = hidden_states[vlayer]
        if cfg.vision_feature_select_strategy == "default":
            x = x[:, 1:]
        with torch.no_grad():
            run_one(interp_block, x, f"block_interp{i}")

    # All spatial blocks (if enabled)
    if cfg.use_spatial_sampling:
        for i in range(4):
            spat_block = model.model.spatial_projectors[i]
            x = hidden_states[cfg.spatial_vision_layer]
            if cfg.vision_feature_select_strategy == "default":
                x = x[:, 1:]
            with torch.no_grad():
                run_one(spat_block, x, f"block_spatial{i}")


def dump_image_newline(model, out_dir: Path):
    save_npy(out_dir / "image_newline.npy", model.model.image_newline.data)


def dump_pack_and_unpad(model, out_dir: Path, cfg):
    """
    Call pack_and_unpad_image_features with a synthetic multi-patch feature
    to validate the unpad + newline insertion logic.
    """
    deterministic()
    # Simulate: original image 600x400 (so num_patches > 1), each patch produces
    # query_side*query_side tokens after downsampling.
    # We do not strictly need matching num_patches; we just need a plausible
    # multi-patch feature whose dimensions follow the HF convention.
    image_size = cfg.vision_config.image_size  # 384
    patch_size = cfg.vision_config.patch_size  # 16
    side = image_size // patch_size  # 24
    # Apply downsample: "4/8" => side * 4/8 = 12
    from fractions import Fraction
    ds = Fraction(cfg.downsample_rate)
    ds_side = int(side * ds)  # 12
    llm_d = cfg.text_config.hidden_size

    # Fake: 1 image, num_patches=3 (base + 2x1 tiles). Image size is exactly
    # 2 tiles tall x 1 tile wide so the aspect matches (no unpad crop).
    # Anyres grid shape: (num_patch_height=2, num_patch_width=1).
    num_patches = 3
    image_features = [torch.randn(num_patches, ds_side * ds_side, llm_d)]
    image_sizes = torch.tensor([[2 * image_size, image_size]])  # (H, W) = (768, 384)

    with torch.no_grad():
        packed, feature_lens = model.model.pack_and_unpad_image_features(
            image_features, image_sizes,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            image_newline=model.model.image_newline,
        )
    save_npy(out_dir / "pack_0_features_in.npy", image_features[0])
    save_npy(out_dir / "pack_0_image_sizes.npy", image_sizes)
    save_npy(out_dir / "pack_0_out.npy", packed[0])

    # Fixture 1: single patch (no splitting) — exercises the short path.
    image_features_1 = [torch.randn(1, ds_side * ds_side, llm_d)]
    image_sizes_1 = torch.tensor([[384, 384]])
    with torch.no_grad():
        packed_1, _ = model.model.pack_and_unpad_image_features(
            image_features_1, image_sizes_1,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            image_newline=model.model.image_newline,
        )
    save_npy(out_dir / "pack_1_features_in.npy", image_features_1[0])
    save_npy(out_dir / "pack_1_image_sizes.npy", image_sizes_1)
    save_npy(out_dir / "pack_1_out.npy", packed_1[0])


def dump_pinpoints(processor, out_dir: Path):
    """For a range of image sizes, record what num_patches and grid shape HF would use."""
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, image_size_to_num_patches,
    )
    sizes = [(384, 384), (600, 400), (400, 600), (1200, 800), (3840, 384), (384, 3840)]
    grid_pinpoints = processor.image_processor.image_grid_pinpoints
    image_size = 384
    out = []
    for h, w in sizes:
        num = image_size_to_num_patches(
            image_size=[h, w], grid_pinpoints=grid_pinpoints, patch_size=image_size,
        )
        npx, npw = get_anyres_image_grid_shape([h, w], grid_pinpoints, image_size)
        out.append({"h": h, "w": w, "num_patches": int(num),
                    "grid": [int(npx), int(npw)]})
    (out_dir / "pinpoints.json").write_text(json.dumps(out, indent=2))


def dump_full_mmproj(model, out_dir: Path, cfg):
    """Run model.model.get_image_features on a canonical image; dump its output (list of streams)."""
    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
    deterministic()
    image_size = cfg.vision_config.image_size
    # Use a deterministic real image size for reproducibility; pick one that
    # triggers anyres splitting (>1 tile).
    target_size = (2 * image_size, image_size)  # (H, W) = (768, 384) → 3 patches total
    num_patches = image_size_to_num_patches(
        image_size=list(target_size),
        grid_pinpoints=cfg.image_grid_pinpoints,
        patch_size=image_size,
    )
    pixel_values = torch.randn(1, num_patches, 3, image_size, image_size)
    image_sizes = torch.tensor([list(target_size)])
    save_npy(out_dir / "mmproj_pixel_values.npy", pixel_values)
    save_npy(out_dir / "mmproj_image_sizes.npy", image_sizes)
    with torch.no_grad():
        features = model.model.get_image_features(
            pixel_values, image_sizes,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
        )
    # features: list of (llm_layer_idx, [tensor per image])
    layer_map = []
    for idx, (llm_layer, packed_list) in enumerate(features):
        save_npy(out_dir / f"mmproj_stream_{idx}_llm_layer_{llm_layer}.npy", packed_list[0])
        layer_map.append({"idx": idx, "llm_layer": int(llm_layer),
                          "n_tokens": int(packed_list[0].shape[0]),
                          "d": int(packed_list[0].shape[1])})
    (out_dir / "mmproj_streams.json").write_text(json.dumps(layer_map, indent=2))


def dump_meta(cfg, out_dir: Path):
    meta = {
        "text": {
            "hidden_size": cfg.text_config.hidden_size,
            "num_hidden_layers": cfg.text_config.num_hidden_layers,
            "num_attention_heads": cfg.text_config.num_attention_heads,
            "num_key_value_heads": cfg.text_config.num_key_value_heads,
            "vocab_size": cfg.text_config.vocab_size,
            "rope_theta": (getattr(cfg.text_config, "rope_theta", None)
                           or cfg.text_config.rope_parameters.get("rope_theta")),
            "logits_scaling": cfg.text_config.logits_scaling,
            "attention_multiplier": cfg.text_config.attention_multiplier,
            "embedding_multiplier": cfg.text_config.embedding_multiplier,
            "residual_multiplier": cfg.text_config.residual_multiplier,
        },
        "vision": {
            "hidden_size": cfg.vision_config.hidden_size,
            "num_hidden_layers": cfg.vision_config.num_hidden_layers,
            "num_attention_heads": cfg.vision_config.num_attention_heads,
            "image_size": cfg.vision_config.image_size,
            "patch_size": cfg.vision_config.patch_size,
            "intermediate_size": cfg.vision_config.intermediate_size,
        },
        "granite4": {
            "downsample_rate": cfg.downsample_rate,
            "deepstack_layer_map": cfg.deepstack_layer_map,
            "use_spatial_sampling": cfg.use_spatial_sampling,
            "spatial_stride": cfg.spatial_stride,
            "spatial_vision_layer": cfg.spatial_vision_layer,
            "spatial_target_layers": cfg.spatial_target_layers,
            "use_image_newline_parameter": cfg.use_image_newline_parameter,
            "image_token_index": cfg.image_token_index,
            "vision_feature_select_strategy": cfg.vision_feature_select_strategy,
            "image_grid_pinpoints": cfg.image_grid_pinpoints,
            "projector_dropout": cfg.projector_dropout,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/proj/mmfm/users/eli/reference/granite-vision-4.1/model")
    ap.add_argument("--out", default="/proj/mmfm/users/eli/reference/granite-vision-4.1/fixtures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Limit to a subset of fixture groups.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, cfg = load_model(args.model_dir)

    groups = args.only or ["meta", "image_newline", "siglip", "downsamplers",
                            "downsampler_substeps", "pack", "pinpoints", "mmproj"]

    if "meta" in groups:
        dump_meta(cfg, out_dir)
    if "image_newline" in groups:
        dump_image_newline(model, out_dir)
    hs = pv = None
    needs_hs = any(g in groups for g in ("siglip", "downsamplers", "downsampler_substeps"))
    if needs_hs:
        hs, pv = dump_siglip(model, out_dir)
    if "downsamplers" in groups:
        dump_downsamplers(model, hs, out_dir, cfg)
    if "downsampler_substeps" in groups:
        dump_downsampler_substeps(model, hs, out_dir, cfg)
    if "pack" in groups:
        dump_pack_and_unpad(model, out_dir, cfg)
    if "pinpoints" in groups:
        dump_pinpoints(processor, out_dir)
    if "mmproj" in groups:
        dump_full_mmproj(model, out_dir, cfg)

    print(f"[done] fixtures -> {out_dir}")


if __name__ == "__main__":
    main()
