#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

/*
 * Granite Vision 4.1 clip graph — WORK IN PROGRESS.
 *
 * Stage 1a (implemented here): the SigLIP vision tower.  build_inp() does
 * the patch + position embedding, and build_vit() runs the 27-layer
 * encoder.  clip_graph::cb() tags every per-layer output ("layer_out-<il>")
 * so the Granite Vision 4.1 fixture harness
 * (tools/mtmd/tests/granite-vision-4.1/test_clip.cpp) can diff each layer
 * against siglip_hidden_states.npy.
 *
 * Stages still to do (see docs/multimodal/granite-vision-4.1.md):
 *
 *   1b. One WindowQFormer block — novel, validated against
 *       fixtures/downsampler_interp_0_{in,out}.npy.
 *
 *   1c. All 8 blocks driven by ctx->model.g4v metadata, each feeding the
 *       hidden state at its projector_vision_layers[bid] and producing a
 *       (n_embd_llm, n_out_tokens) stream.  Validate against
 *       fixtures/mmproj_stream_{0..7}_llm_layer_*.npy.
 *
 * Until 1c lands, this file returns the final SigLIP hidden state as the
 * graph output so clip.cpp's post-compute assertions can run.  That path
 * goes through the mmproj's copy of v.post_ln; Granite Vision 4.1's real
 * projectors consume the *pre-post-ln* hidden states, but the per-layer
 * captures taken by cb_eval are pre-post-ln already, so diffing is still
 * correct.  clip_n_output_tokens / clip_n_mmproj_embd must therefore
 * report the encoder's raw output shape (576, 1152) for this stage; both
 * will switch to (n_out_tokens, projection_dim) when 1c lands.
 */
ggml_cgraph * clip_graph_granite4v::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr && "granite4v: missing v.patch_embd.weight");
    GGML_ASSERT(model.position_embeddings != nullptr && "granite4v: missing v.position_embd.weight");
    GGML_ASSERT(model.class_embedding == nullptr && "granite4v: SigLIP tower should not have a CLS token");

    ggml_tensor * inp = build_inp();  // conv2d + patch_bias -> (n_embd, n_patches)

    // SigLIP encoder.  eps, ffn_op (= FFN_GELU, the tanh variant, matching
    // HF's gelu_pytorch_tanh), n_head, n_layer all come from model.hparams.
    ggml_tensor * cur = build_vit(
        inp,
        /* n_pos            */ n_patches,
        /* norm_t           */ NORM_TYPE_NORMAL,
        /* ffn_t            */ hparams.ffn_op,
        /* learned_pos_embd */ model.position_embeddings,
        /* add_pos          */ nullptr);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
