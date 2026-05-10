#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

/*
 * Granite Vision 4.1 clip graph — WORK IN PROGRESS.
 *
 *   Stage 1a: SigLIP vision tower (27 layers, post-norm) — done.
 *   Stage 1b: one WindowQFormer block (bid=0, interp) — done here.
 *             Validates against fixtures/block_interp0_{norm,enc,
 *             downsampled,query_embeds,encoder_embeds,qformer_out,
 *             unwin,out}.npy via cb_eval capture names "g4v_blk0_*".
 *   Stage 1c: dispatch over all 8 blocks using ctx->model.g4v metadata.
 *
 *   A single WindowQFormer block maps a (1, 576, 1152) vision-layer
 *   hidden state h to a (1, 144, 2560) deepstack feature:
 *
 *     x      = LN(h)                                       # (1, 576, 1152)
 *     enc    = _win(x, side=24, win=8)                     # (9, 64, 1152)
 *     d      = interp(x, 24->12) or spatial_offset(x, 24->12, off)
 *                                                           # (1, 144, 1152)
 *     q_in   = query + _win(d, side=12, win=4)             # (9, 16, 1152)
 *     e_in   = enc + image_positions                       # (9, 64, 1152)
 *     # Qformer forward (post-norm Bert style):
 *     q_in   = LN_embed(q_in)                              # per-query LN
 *     sa_out = LN_sa(dense_sa(MHA(q_in,q_in,q_in)) + q_in)
 *     ca_out = LN_ca(dense_ca(MHA(sa_out,e_in,e_in)) + sa_out)
 *     ff_out = LN_ffn(dense_down(gelu_erf(dense_up(ca_out))) + ca_out)
 *     out    = out_linear(_unwin(ff_out, n=3, win=4))       # (1, 144, 2560)
 *
 *   Notes:
 *   - Qformer FFN uses HF's default "gelu" = erf variant (not tanh).
 *   - Qformer LN eps = 1e-12 (from Blip2QFormerConfig).
 *   - Dropout is inference-time no-op.
 *   - Sub-step tagging via cb() allows the fixture harness to diff each
 *     intermediate against its .npy counterpart without needing a
 *     partial-graph API.
 */

// ---------------------------------------------------------------------------
// Permutation helpers
// ---------------------------------------------------------------------------
//
// _win(side, win) reshapes a (1, side^2, C) raster tensor to (n^2, win^2, C)
// where n = side/win and windows are raster-ordered row-major, as are the
// tokens inside each window.  In ggml the natural way to express this
// permutation is with ggml_get_rows over a precomputed index tensor.
//
// _unwin and spatial-offset sampling are also permutations.
//
// The index tensors are created as graph inputs with stable names; their
// values are populated during clip_image_batch_encode (see the GRANITE4V
// case in clip.cpp), so both definitions of the permutation must stay in
// sync.

// ---------------------------------------------------------------------------
// Block forward helpers
// ---------------------------------------------------------------------------

// Apply ggml_get_rows with a named input index tensor; returns a
// (C, idx_len, ...) tensor.  The index tensor is created as a graph input
// so the harness / clip.cpp can upload its contents.
static ggml_tensor * g4v_gather(ggml_context * ctx, ggml_cgraph * gf,
                                ggml_tensor * src,
                                const std::string & name,
                                int idx_len) {
    (void) gf;
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, idx_len);
    ggml_set_name(idx, name.c_str());
    ggml_set_input(idx);
    return ggml_get_rows(ctx, src, idx);
}

// Area-interpolate 24x24 -> 12x12 via 2x2 avg pool.  src shape (C, 576) with
// axis 1 = raster (row-major, (y*side + x)).  Output shape (C, 144) raster
// row-major.
//
// ggml_permute(a, a0, a1, a2, a3): new.ne[a_i] = old.ne[i].  So to get
// (x, y, C, 1) from (C, x, y, 1) we map old[0]=C → new[2], old[1]=x → new[0],
// old[2]=y → new[1], old[3]=1 → new[3], which is permute(t, 2, 0, 1, 3).
static ggml_tensor * g4v_interp_down(ggml_context * ctx, ggml_tensor * src, int side, int new_side) {
    const int n_embd = src->ne[0];
    // Reshape (C, side^2) -> (C, side, side, 1) with ne[1]=x, ne[2]=y.
    ggml_tensor * t = ggml_reshape_4d(ctx, src, n_embd, side, side, 1);
    // Permute to (x, y, C, 1) for ggml_pool_2d (pools ne[0] and ne[1]).
    t = ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));
    const int kernel = side / new_side;
    t = ggml_pool_2d(ctx, t, GGML_OP_POOL_AVG, kernel, kernel, kernel, kernel, 0, 0);
    // Now t shape (new_side, new_side, C, 1).  Permute back to
    // (C, new_side, new_side, 1): old[0]=x → new[1], old[1]=y → new[2],
    // old[2]=C → new[0] → permute(t, 1, 2, 0, 3).
    t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));
    return ggml_reshape_2d(ctx, t, n_embd, new_side * new_side);
}

// Build one WindowQFormer block's forward pass.  Returns a tensor of shape
// (D_llm, query_side^2 * n^2) in ggml ne order = (2560, 144) for block 0.
//
// cb_prefix is used to tag intermediates (e.g. "g4v_blk0") so the fixture
// harness can validate each sub-step.
static ggml_tensor * g4v_build_block(
        const clip_graph * g,
        ggml_context * ctx,
        ggml_cgraph * gf,
        const clip_model::g4v_projector_block & blk,
        ggml_tensor * h,                 // (C=1152, 576) raster
        int bid,
        bool is_spatial,
        int spatial_offset,
        int image_side,
        int window_side,
        int query_side,
        float qformer_eps) {

    const int n_embd = h->ne[0];
    GGML_ASSERT(h->ne[1] == image_side * image_side);
    const int n = image_side / window_side;
    const int new_side = n * query_side;
    const int n_windows = n * n;
    const int enc_len = window_side * window_side;
    const int query_len = query_side * query_side;

    // Tag a tensor with a stable name and make sure it shows up in the
    // scheduler's cb_eval stream.  A ggml_cont produces a fresh, node-level
    // tensor that cannot be folded into a view, so the scheduler's observe
    // callback will fire for it.  Downstream ops take the cont tensor so
    // the whole graph still flows through a named point.
    auto cbx = [&](ggml_tensor * & t, const char * step) {
        const std::string name = "g4v_blk" + std::to_string(bid) + "_" + step;
        ggml_tensor * tagged = ggml_cont(ctx, t);
        ggml_set_name(tagged, name.c_str());
        ggml_build_forward_expand(gf, tagged);
        t = tagged; // downstream ops see the tagged tensor
    };

    // 1. Top-level LN
    ggml_tensor * x = g->build_norm(h, blk.norm_w, blk.norm_b, NORM_TYPE_NORMAL, g->eps, bid);
    cbx(x, "norm");

    // 2. enc = _win(x, image_side, window_side) → (C, enc_len, n_windows)
    ggml_tensor * enc;
    {
        ggml_tensor * enc_flat = g4v_gather(ctx, gf, x,
            "g4v_blk" + std::to_string(bid) + "_win_idx",
            image_side * image_side);
        enc = ggml_reshape_3d(ctx, enc_flat, n_embd, enc_len, n_windows);
    }
    cbx(enc, "enc");

    // 3. downsampled = downsampler(x) → (C, new_side^2)
    ggml_tensor * d;
    (void) spatial_offset; // set via index upload, not at build-time
    if (is_spatial) {
        // Spatial-offset: pick one of 4 positions from each 2x2 block.
        d = g4v_gather(ctx, gf, x,
            "g4v_blk" + std::to_string(bid) + "_spatial_idx",
            new_side * new_side);
    } else {
        d = g4v_interp_down(ctx, x, image_side, new_side);
    }
    cbx(d, "downsampled");

    // 4. query_embeds = query + _win(d, new_side, query_side) → (C, query_len, n_windows)
    ggml_tensor * q_in;
    {
        ggml_tensor * dw_flat = g4v_gather(ctx, gf, d,
            "g4v_blk" + std::to_string(bid) + "_qwin_idx",
            new_side * new_side);
        ggml_tensor * dw = ggml_reshape_3d(ctx, dw_flat, n_embd, query_len, n_windows);
        // blk.query has HF shape (1, query_len, C) → ggml ne (C, query_len, 1).
        // Broadcast-add over the n_windows axis.
        q_in = ggml_add(ctx, dw, blk.query);
    }
    cbx(q_in, "query_embeds");

    // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
    ggml_tensor * e_in = ggml_add(ctx, enc, blk.image_positions);
    cbx(e_in, "encoder_embeds");

    // 6. Qformer forward.  Per Blip2QFormerModel.forward: first runs
    //    self.layernorm(query_embeds) before entering the single layer.
    //    blk.post_norm_* holds this (our naming is historical; see the
    //    comment in clip-model.h's g4v_projector_block struct).
    ggml_tensor * q = g->build_norm(q_in, blk.post_norm_w, blk.post_norm_b, NORM_TYPE_NORMAL, qformer_eps, bid);

    // Helper: one post-norm BERT attention block.
    //   attn(q,k,v) via dense W^Q/W^K/W^V + scaled dot-product
    //   residual: LN_out(dropout(dense(attn_out)) + residual_input)
    // For self-attention q == k == v == residual. For cross-attn q != k=v.
    auto run_postnorm_attn = [&](
            ggml_tensor * q_stream,   // (C, nq, n_win) Q source and residual
            ggml_tensor * kv_stream,  // (C, nkv, n_win) K/V source
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * wo, ggml_tensor * bo,
            ggml_tensor * ln_w, ggml_tensor * ln_b) -> ggml_tensor * {
        const int n_head = 18;                 // QFormer: 1152 / 64
        const int d_h   = n_embd / n_head;
        const int nq   = q_stream->ne[1];
        const int nkv  = kv_stream->ne[1];

        // We loop over the n_windows dimension by treating it as batch via
        // the ne[2] dim in mul_mat.  Collapse (C, n, n_win) → (C, n * n_win)
        // for the linear projections (they don't care about the window dim),
        // then reshape back.
        auto linear = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
            ggml_tensor * t = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
            t = g->build_mm(w, t);
            if (b) t = ggml_add(ctx, t, b);
            return t;
        };

        ggml_tensor * Q = linear(q_stream, wq, bq);    // (C, nq * n_win)
        ggml_tensor * K = linear(kv_stream, wk, bk);   // (C, nkv * n_win)
        ggml_tensor * V = linear(kv_stream, wv, bv);   // (C, nkv * n_win)

        // Reshape for attention.  Shape: (d_h, n_head, n_pos, n_win).
        Q = ggml_reshape_4d(ctx, Q, d_h, n_head, nq,  n_windows);
        K = ggml_reshape_4d(ctx, K, d_h, n_head, nkv, n_windows);
        V = ggml_reshape_4d(ctx, V, d_h, n_head, nkv, n_windows);

        // Permute to (d_h, n_pos, n_head, n_win) for mul_mat.
        ggml_tensor * q_p = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor * k_p = ggml_permute(ctx, K, 0, 2, 1, 3);
        // For V we need (n_pos, d_h, n_head, n_win) so that V @ kq gives
        // (d_h, nq, n_head, n_win).
        ggml_tensor * v_p = ggml_permute(ctx, V, 1, 2, 0, 3);
        v_p = ggml_cont(ctx, v_p);

        // kq = K @ Q  → (nkv, nq, n_head, n_win) after mul_mat(K_p, q_p).
        // ggml_mul_mat(A, B): A(ne0, nA, ..) × B(ne0, nB, ..) = (nA, nB, ..)
        // inner dim must match ne[0].
        ggml_tensor * kq = ggml_mul_mat(ctx, k_p, q_p);
        const float scale = 1.0f / std::sqrt((float) d_h);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        // kq: (nkv, nq, n_head, n_win)

        // ctx = V @ kq → V_p(nkv, d_h, n_head, n_win) × kq(nkv, nq, ...) = (d_h, nq, n_head, n_win)
        ggml_tensor * kqv = ggml_mul_mat(ctx, v_p, kq);
        // Permute back to (d_h, n_head, nq, n_win) and cont → (n_embd, nq, n_win).
        ggml_tensor * out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        out = ggml_cont_3d(ctx, out, d_h * n_head, nq, n_windows);

        // Output projection (linear) + residual + LN_out (post-norm).
        ggml_tensor * o = ggml_reshape_2d(ctx, out, n_embd, nq * n_windows);
        o = g->build_mm(wo, o);
        if (bo) o = ggml_add(ctx, o, bo);
        o = ggml_reshape_3d(ctx, o, n_embd, nq, n_windows);
        // Residual
        o = ggml_add(ctx, o, q_stream);
        // Post-norm
        o = g->build_norm(o, ln_w, ln_b, NORM_TYPE_NORMAL, qformer_eps, bid);
        return o;
    };

    // 6a. Self-attention on q.
    ggml_tensor * sa_out = run_postnorm_attn(
        q, q,
        blk.sa_q_w,  blk.sa_q_b,
        blk.sa_k_w,  blk.sa_k_b,
        blk.sa_v_w,  blk.sa_v_b,
        blk.sa_out_w, blk.sa_out_b,
        blk.sa_out_ln_w, blk.sa_out_ln_b);
    cbx(sa_out, "sa_out");

    // 6b. Cross-attention: Q from sa_out, K/V from e_in.
    ggml_tensor * ca_out = run_postnorm_attn(
        sa_out, e_in,
        blk.ca_q_w,  blk.ca_q_b,
        blk.ca_k_w,  blk.ca_k_b,
        blk.ca_v_w,  blk.ca_v_b,
        blk.ca_out_w, blk.ca_out_b,
        blk.ca_out_ln_w, blk.ca_out_ln_b);
    cbx(ca_out, "ca_out");

    // 6c. FFN: gelu_erf(up(x)) → down → + residual → post-norm.
    //    HF's ACT2FN["gelu"] is the erf variant.
    ggml_tensor * ffn;
    {
        ggml_tensor * t = ggml_reshape_2d(ctx, ca_out, n_embd, query_len * n_windows);
        t = g->build_mm(blk.ffn_up_w, t);
        if (blk.ffn_up_b) t = ggml_add(ctx, t, blk.ffn_up_b);
        t = ggml_gelu_erf(ctx, t);
        t = g->build_mm(blk.ffn_down_w, t);
        if (blk.ffn_down_b) t = ggml_add(ctx, t, blk.ffn_down_b);
        t = ggml_reshape_3d(ctx, t, n_embd, query_len, n_windows);
        ffn = ggml_add(ctx, t, ca_out);
        ffn = g->build_norm(ffn, blk.ffn_ln_w, blk.ffn_ln_b, NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ffn, "qformer_out");

    // 7. _unwin back to raster (C, new_side^2)
    ggml_tensor * unwinned;
    {
        // ffn is (C, query_len, n_windows) in window-major order.  Flatten
        // to (C, query_len * n_windows) = (C, new_side^2); unwin_idx
        // reorders those tokens into raster (y, x) order over new_side.
        ggml_tensor * flat = ggml_reshape_2d(ctx, ffn, n_embd, query_len * n_windows);
        unwinned = g4v_gather(ctx, gf, flat,
            "g4v_blk" + std::to_string(bid) + "_unwin_idx",
            new_side * new_side);
    }
    cbx(unwinned, "unwin");

    // 8. out_linear
    ggml_tensor * out = g->build_mm(blk.out_linear_w, unwinned);
    if (blk.out_linear_b) out = ggml_add(ctx, out, blk.out_linear_b);
    cbx(out, "out");

    return out;
}

// ---------------------------------------------------------------------------
// build() — top-level graph
// ---------------------------------------------------------------------------

ggml_cgraph * clip_graph_granite4v::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(!model.g4v.blocks.empty());

    // --- Stage 1a: SigLIP encoder producing intermediate hidden states. ---
    // We cannot use build_vit() directly because the projector bank pulls
    // from multiple intermediate layers; duplicate its loop and stash the
    // per-layer outputs we need in a side vector.
    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "pos_embed", -1);

    ggml_tensor * inpL = inp;
    std::vector<ggml_tensor *> layer_outs(n_layer, nullptr);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        // Self-attention (fused qkv unsupported for SigLIP of this shape; all sep)
        ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
        ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
        ggml_tensor * Vcur = build_mm(layer.v_w, cur);
        if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
        Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

        cur = build_attn(layer.o_w, layer.o_b,
                         Qcur, Kcur, Vcur, nullptr, kq_scale, il);

        cur = ggml_add(ctx0, cur, inpL); // residual 1
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
                        layer.ff_up_w, layer.ff_up_b,
                        layer.ff_gate_w, layer.ff_gate_b,
                        layer.ff_down_w, layer.ff_down_b,
                        hparams.ffn_op, il);
        cur = ggml_add(ctx0, inpL, cur); // residual 2
        cb(cur, "layer_out", il);
        layer_outs[il] = cur;
        inpL = cur;
    }

    // --- Stages 1c + 2a: all 8 WindowQFormer blocks, packed into the
    //                     final mmproj output.
    //
    // Each block pulls from its own SigLIP vision layer (stored as a
    // negative HF index in projector_vision_layers) and produces a
    // (D_llm, query_side^2 * n^2) = (2560, 144) output stream.  All 8
    // streams are then:
    //   (1) sorted by projector_llm_layers[bid] ascending, so the stream
    //       targeting the lowest decoder layer (= the "base" vision
    //       feature in qwen3vl-deepstack parlance) lands at feature
    //       offset 0, and successive streams at (k * n_embd).  Granite
    //       Vision 4.1's HF forward zeroes image-position inputs_embeds
    //       and adds each stream at its target layer, so the stream
    //       with llm_layer=0 effectively IS the base.
    //   (2) concatenated along the feature dim into a single tensor of
    //       shape (8 * projection_dim, 144).
    //   (3) appended with one image_newline token along the token dim
    //       (HF pack_and_unpad_image_features single-tile branch); each
    //       stream slice of the newline token equals the learned
    //       image_newline vector.
    //
    // Final output shape: (8 * projection_dim, 144 + 1) = (20480, 145).
    //
    // Multi-tile (anyres) inputs will be handled at the mtmd layer by
    // calling clip encoder once per tile and doing the pack_and_unpad
    // assembly there; the single-tile path below matches HF
    // pack_and_unpad's `else` branch exactly.
    const auto & g4v = model.g4v;
    const float qformer_eps = 1e-12f;  // Blip2QFormerConfig default
    const int   projection_dim = hparams.projection_dim;

    std::vector<ggml_tensor *> streams(g4v.projector_count, nullptr);
    for (int bid = 0; bid < g4v.projector_count; ++bid) {
        const auto & blk = g4v.blocks[bid];

        int vlayer = g4v.projector_vision_layers[bid];
        if (vlayer < 0) vlayer = (n_layer + 1) + vlayer;  // HF negative-indexing
        GGML_ASSERT(vlayer >= 1 && vlayer <= n_layer);
        ggml_tensor * h = layer_outs[vlayer - 1];

        streams[bid] = g4v_build_block(
            this, ctx0, gf, blk,
            h, bid,
            g4v.projector_is_spatial[bid],
            g4v.projector_spatial_offset[bid],
            /* image_side  */ n_patches_x,
            /* window_side */ g4v.downsample_window_side,
            /* query_side  */ g4v.downsample_query_side,
            qformer_eps);
    }

    // Sort block ids by projector_llm_layers ascending.
    std::vector<int> order(g4v.projector_count);
    for (int i = 0; i < g4v.projector_count; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return g4v.projector_llm_layers[a] < g4v.projector_llm_layers[b];
    });

    // Concatenate streams along the feature dim (ne[0]).  Each stream has
    // shape (projection_dim, 144).  Result: (K * projection_dim, 144).
    ggml_tensor * mmproj = nullptr;
    for (int k = 0; k < g4v.projector_count; ++k) {
        ggml_tensor * s = streams[order[k]];
        mmproj = (k == 0) ? s : ggml_concat(ctx0, mmproj, s, /*dim=*/0);
    }
    ggml_set_name(mmproj, "g4v_mmproj_packed");

    // Append the learned image_newline vector as one extra token along the
    // token dim (ne[1]).  image_newline has shape (projection_dim,) and is
    // replicated across all K stream slices.
    const int n_tokens_base = static_cast<int>(streams[0]->ne[1]);
    const int n_streams     = g4v.projector_count;
    ggml_tensor * newline_token;
    {
        // Tile image_newline K times along ne[0] to build one (K*D, 1)
        // token, then concat along ne[1].  ggml_repeat expands the tensor
        // to a target shape; we use an explicit new tensor of the target
        // shape to drive it.
        GGML_ASSERT(model.image_newline != nullptr);
        ggml_tensor * newline = model.image_newline;         // (D,)
        newline = ggml_reshape_2d(ctx0, newline, projection_dim, 1);  // (D, 1)
        ggml_tensor * target = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                                  n_streams * projection_dim, 1);
        newline_token = ggml_repeat(ctx0, newline, target);  // (K*D, 1)
    }
    mmproj = ggml_concat(ctx0, mmproj, newline_token, /*dim=*/1);
    ggml_set_name(mmproj, "g4v_mmproj_out");

    (void) n_tokens_base;  // referenced only for readability above

    ggml_build_forward_expand(gf, mmproj);
    return gf;
}
