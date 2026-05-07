#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully
 * or predominantly AI-generated.  We encourage human contributors to
 * ensure the quality and reliability of the codebase.
 *
 * Granite Vision 4.1 clip graph — WORK IN PROGRESS.
 *
 * This file deliberately aborts at build-time until the WindowQFormer
 * forward pass is implemented.  The ctx->vit_model has already been
 * populated with the SigLIP encoder tensors and a fully-loaded g4v
 * projector bank (see load_tensors in clip.cpp).  The author's task is
 * to:
 *
 *   1. Build the SigLIP encoder cgraph up to the stack of hidden states
 *      needed by the 8 projector blocks (vision layers -19/-13/-7/-1 and
 *      the spatial_vision_layer copy of the last layer).
 *
 *   2. For each projector block bid ∈ {0..g4v.projector_count-1}:
 *        a. Read the corresponding hidden state h_v from the tower.
 *        b. If the strategy is "default", drop CLS (not present in SigLIP,
 *           so no-op in practice).
 *        c. Apply the block's norm, then produce enc = _win(h_v, image_side,
 *           window_side) of shape (B*n*n, w*w, D_v).
 *        d. Downsample h_v to produce the query initialization:
 *             - interpolate (area) if !is_spatial[bid]
 *             - spatial offset sampling if is_spatial[bid]
 *        e. Build query_embeds = block.query + _win(downsampled, new_side,
 *           query_side) and encoder_embeds = enc + block.image_positions.
 *        f. Run a single BLIP-2 QFormer layer (self-attn, cross-attn, FFN)
 *           using block.sa_*, block.ca_*, block.ffn_*.
 *        g. Apply block.post_norm on the output, _unwin to (B, (n*q)^2, D_v),
 *           and project through block.out_linear to D_llm.
 *
 *   3. Return the list of 8 projector outputs as the graph result.  The
 *      caller (mtmd) will pack them plus image_newline into the final
 *      multi-stream embedding.
 *
 * Fixtures to validate against live under
 * /proj/mmfm/users/eli/reference/granite-vision-4.1/fixtures:
 *   - siglip_input.npy, siglip_hidden_states.npy
 *   - downsampler_interp_<0..3>_in.npy / _out.npy
 *   - downsampler_spatial_<0..3>_in.npy / _out.npy
 *   - mmproj_pixel_values.npy, mmproj_image_sizes.npy,
 *     mmproj_stream_<0..7>_llm_layer_<N>.npy
 *
 * See docs/multimodal/granite-vision-4.1.md for the full design.
 */
ggml_cgraph * clip_graph_granite4v::build() {
    GGML_ABORT("clip_graph_granite4v::build not yet implemented — see "
               "docs/multimodal/granite-vision-4.1.md");
}
