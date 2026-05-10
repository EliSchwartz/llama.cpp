// Granite Vision 4.1 clip-graph fixture harness.
//
// Loads the converted mmproj.gguf, feeds the SigLIP pixel fixture through
// clip_encode_float_image, and diffs named intermediate tensors against
// the .npy fixtures produced by gen_fixtures.py.
//
// The harness drives clip.cpp through its public API only.  It installs a
// ggml_backend_sched_eval_callback and snapshots any tensor whose name
// matches a requested checkpoint.  As the granite4v forward graph is
// filled in (SigLIP encoder, then WindowQFormer, then the 8-block loop),
// each stage's output becomes diffable from here — nothing else about the
// harness needs to change.
//
// Usage:
//   test_clip --mmproj <path> --fixtures <dir>
//
// If the graph is not yet implemented the harness surfaces the abort from
// clip_graph_granite4v::build() and reports which checkpoints never
// flowed through cb_eval.

#include "clip.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct checkpoint {
    std::string name;    // ggml tensor name set by clip_graph::cb()
    std::string fixture; // path to .npy with the expected values
};

struct harness {
    // Requested checkpoints, keyed by tensor name.
    std::map<std::string, std::string> wanted;
    // Captured values after graph compute, keyed by tensor name.
    std::map<std::string, std::vector<float>> captured;
};

// Set just before clip_encode_float_image and consumed by the eval callback.
// The callback has no user pointer wired through clip.h, so we pass via a
// file-scope variable.  Safe because the harness is single-threaded.
harness * g_harness = nullptr;

bool eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_harness) return true;

    const char * raw_name = ggml_get_name(t);
    if (!raw_name || raw_name[0] == '\0') return false;

    // Debug: dump every name the scheduler offers us, plus whether it's
    // ask/observe.  Enable with G4V_DEBUG_NAMES=1.
    if (std::getenv("G4V_DEBUG_NAMES")) {
        std::fprintf(stderr, "cb_eval %s: %s\n", ask ? "ask " : "obs ", raw_name);
    }

    auto it = g_harness->wanted.find(raw_name);
    if (it == g_harness->wanted.end()) return false;

    if (ask) {
        // Tell the scheduler: yes, we want to observe this node.
        return true;
    }

    const int64_t n = ggml_nelements(t);
    std::vector<float> buf(static_cast<size_t>(n));
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    g_harness->captured[raw_name] = std::move(buf);
    return true; // do not cancel compute
}

// Transpose a (3, H, W) CHW float array into HWC (RGB interleaved), as
// expected by clip_encode_float_image (see clip.cpp::clip_image_batch_encode
// which immediately re-interleaves it back to CHW).
std::vector<float> chw_to_hwc(const std::vector<float> & chw, int64_t h, int64_t w) {
    std::vector<float> hwc(static_cast<size_t>(3 * h * w));
    const int64_t plane = h * w;
    for (int64_t y = 0; y < h; y++) {
        for (int64_t x = 0; x < w; x++) {
            const int64_t dst = 3 * (y * w + x);
            const int64_t src = y * w + x;
            hwc[dst    ] = chw[0 * plane + src];
            hwc[dst + 1] = chw[1 * plane + src];
            hwc[dst + 2] = chw[2 * plane + src];
        }
    }
    return hwc;
}

int run(const std::string & mmproj, const std::string & fixtures_dir) {
    // Checkpoints we expect to capture once the SigLIP encoder is in place.
    // Each entry names a ggml tensor (set by clip_graph::cb()) and the .npy
    // slice it should match.  The SigLIP stack has 27 layers and fixture
    // siglip_hidden_states[i] corresponds to:
    //   i = 0        -> post patch-embed + pos (pre-encoder), ggml name "pos_embed"
    //   i = 1..27    -> output of transformer layer i-1,      ggml name "layer_out-<i-1>"
    //
    // All 8 blocks pull from layers -19/-13/-7/-1 (interp) and -1 (spatial),
    // so with stage 1c the full 27-layer stack is reachable.  Sample a
    // handful of layers to catch divergence anywhere in the tower.
    std::vector<checkpoint> checkpoints = {
        {"pos_embed",    "siglip_hidden_states_layer_0.npy"},
        {"layer_out-0",  "siglip_hidden_states_layer_1.npy"},
        {"layer_out-8",  "siglip_hidden_states_layer_9.npy"},   // bid 0 vision_layer
        {"layer_out-14", "siglip_hidden_states_layer_15.npy"},  // bid 1 vision_layer
        {"layer_out-20", "siglip_hidden_states_layer_21.npy"},  // bid 2 vision_layer
        {"layer_out-26", "siglip_hidden_states_layer_27.npy"},  // bid 3/spatial vision_layer
    };

    // The fixture is a single (28, 1, 576, 1152) file.  Rather than pre-slice
    // it on disk, we load the full array once and memcmp against the right
    // stride for each checkpoint.
    g4v_npy::array_f32 hs;
    try {
        hs = g4v_npy::load_f32(fixtures_dir + "/siglip_hidden_states.npy");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: could not load siglip_hidden_states.npy: %s\n", e.what());
        return 2;
    }
    if (hs.shape.size() != 4 || hs.shape[0] != 28 || hs.shape[1] != 1
        || hs.shape[2] != 576 || hs.shape[3] != 1152) {
        std::fprintf(stderr, "error: unexpected siglip_hidden_states shape\n");
        return 2;
    }
    const int64_t layer_stride = hs.shape[1] * hs.shape[2] * hs.shape[3];

    g4v_npy::array_f32 pixel;
    try {
        pixel = g4v_npy::load_f32(fixtures_dir + "/siglip_input.npy");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: could not load siglip_input.npy: %s\n", e.what());
        return 2;
    }
    if (pixel.shape.size() != 4 || pixel.shape[0] != 1 || pixel.shape[1] != 3
        || pixel.shape[2] != 384 || pixel.shape[3] != 384) {
        std::fprintf(stderr, "error: unexpected siglip_input shape\n");
        return 2;
    }

    harness h;
    for (const auto & cp : checkpoints) {
        h.wanted.emplace(cp.name, cp.fixture);
    }
    // Block sub-step checkpoints.  These must be registered in h.wanted
    // BEFORE clip_encode_float_image runs, otherwise cb_eval returns false
    // during graph compute and the tensors never get observed.
    //
    // Block 0 (interp) gets full per-sub-step coverage; blocks 1-7 only
    // check the final "out" fixture, which implicitly covers everything
    // upstream of it.  Add more sub-step fixtures to block 4 (first
    // spatial) if spatial-specific debugging is ever needed — the existing
    // block_spatial0_*.npy fixtures are already on disk.
    struct block_check { const char * name; const char * fixture; };
    const std::vector<block_check> block_checks = {
        {"g4v_blk0_norm",           "block_interp0_norm.npy"},
        {"g4v_blk0_enc",            "block_interp0_enc.npy"},
        {"g4v_blk0_downsampled",    "block_interp0_downsampled.npy"},
        {"g4v_blk0_query_embeds",   "block_interp0_query_embeds.npy"},
        {"g4v_blk0_encoder_embeds", "block_interp0_encoder_embeds.npy"},
        {"g4v_blk0_qformer_out",    "block_interp0_qformer_out.npy"},
        {"g4v_blk0_unwin",          "block_interp0_unwin.npy"},
        {"g4v_blk0_out",            "block_interp0_out.npy"},
        // Block 1 sub-steps to isolate the divergence in the failing blocks.
        {"g4v_blk1_norm",           "block_interp1_norm.npy"},
        {"g4v_blk1_enc",            "block_interp1_enc.npy"},
        {"g4v_blk1_downsampled",    "block_interp1_downsampled.npy"},
        {"g4v_blk1_query_embeds",   "block_interp1_query_embeds.npy"},
        {"g4v_blk1_qformer_out",    "block_interp1_qformer_out.npy"},
        // Per-block final outputs; bid 0..3 = interp, bid 4..7 = spatial.
        {"g4v_blk1_out",            "downsampler_interp_1_out.npy"},
        {"g4v_blk2_out",            "downsampler_interp_2_out.npy"},
        {"g4v_blk3_out",            "downsampler_interp_3_out.npy"},
        {"g4v_blk4_out",            "downsampler_spatial_0_out.npy"},
        {"g4v_blk5_out",            "downsampler_spatial_1_out.npy"},
        {"g4v_blk6_out",            "downsampler_spatial_2_out.npy"},
        {"g4v_blk7_out",            "downsampler_spatial_3_out.npy"},
        // Stage 2a: full mmproj output = 8 streams sorted by llm_layer
        // ascending, concatenated along feature dim, with one appended
        // image_newline token.  Shape (145, 8*2560) in PyTorch / memory,
        // (8*2560, 145) in ggml ne order.
        {"g4v_mmproj_out",          "mmproj_out_single_tile.npy"},
    };
    for (const auto & bc : block_checks) {
        h.wanted.emplace(bc.name, bc.fixture);
    }
    g_harness = &h;

    clip_context_params cparams = {};
    cparams.use_gpu           = false;
    cparams.flash_attn_type   = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cparams.image_min_tokens  = -1;
    cparams.image_max_tokens  = -1;
    cparams.warmup            = false;  // don't run the graph at init; it may not be ready
    cparams.cb_eval           = eval_cb;
    cparams.cb_eval_user_data = nullptr;

    clip_init_result init = clip_init(mmproj.c_str(), cparams);
    if (!init.ctx_v) {
        std::fprintf(stderr, "error: clip_init returned null vision context for %s\n", mmproj.c_str());
        g_harness = nullptr;
        return 3;
    }

    const int n_threads = 4;
    std::vector<float> hwc = chw_to_hwc(pixel.data, 384, 384);
    // clip_n_mmproj_embd() is the base per-token size; a granite4v mmproj
    // produces (1 + n_deepstack) * n_embd floats per token.  Oversize the
    // destination buffer to avoid writing past the end.  The harness does
    // not use the final embedding — the diff is done on captured nodes.
    const size_t out_cap = 4 * 1024 * 1024; // 16 MB of floats
    std::vector<float> out(out_cap);

    std::fprintf(stdout, "running clip_encode_float_image...\n");
    std::fflush(stdout);
    bool ok = clip_encode_float_image(init.ctx_v, n_threads, hwc.data(), 384, 384, out.data());
    std::fprintf(stdout, "clip_encode_float_image returned %s\n", ok ? "true" : "false");

    int failed = 0;
    for (const auto & cp : checkpoints) {
        auto it = h.captured.find(cp.name);
        if (it == h.captured.end()) {
            std::fprintf(stdout, "  [MISS] %-26s : tensor never flowed through cb_eval\n", cp.name.c_str());
            failed++;
            continue;
        }

        // Map the checkpoint's fixture filename back to a layer index; for
        // the SigLIP stack we compare against siglip_hidden_states[layer].
        int layer = -1;
        if (cp.name == "pos_embed") {
            layer = 0;
        } else if (cp.name.rfind("layer_out-", 0) == 0) {
            layer = std::atoi(cp.name.c_str() + std::strlen("layer_out-")) + 1;
        }
        if (layer < 0 || layer >= hs.shape[0]) {
            std::fprintf(stdout, "  [SKIP] %-26s : no fixture mapping\n", cp.name.c_str());
            continue;
        }
        const float * expected = hs.data.data() + layer * layer_stride;
        const std::vector<float> & got = it->second;
        if (static_cast<int64_t>(got.size()) != layer_stride) {
            std::fprintf(stdout, "  [SIZE] %-26s : got %zu floats, expected %lld\n",
                         cp.name.c_str(), got.size(), static_cast<long long>(layer_stride));
            failed++;
            continue;
        }
        auto rep = g4v_npy::diff_stats(got.data(), expected, static_cast<size_t>(layer_stride));
        // Tolerance rationale: ggml CPU uses an FP16 lookup table for gelu
        // (GGML_GELU_FP16 in ggml/src/ggml-cpu/vec.h), which introduces
        // ~1e-3 relative error per FFN call.  Over 27 layers this compounds
        // at a handful of attention-sink positions but stays small in
        // mean.  We check the max diff against the largest reference
        // activation magnitude — a proper relative bound — rather than
        // raw max_abs.
        const double rel_to_ref_max = rep.ref_p99 > 0
            ? rep.max_abs / rep.ref_p99
            : rep.max_abs;
        const bool pass = rel_to_ref_max <= 5e-4 && rep.mean_abs <= 5e-3;
        std::fprintf(stdout,
                     "  [%s] %-26s : max_abs=%.4g  mean_abs=%.4g  max_abs/ref_max=%.4g  ref_max=%.4g\n",
                     pass ? " OK " : "FAIL", cp.name.c_str(),
                     rep.max_abs, rep.mean_abs, rel_to_ref_max, rep.ref_p99);
        if (!pass) failed++;
    }

    // Block sub-step diffs.  Each checks the captured tensor against a
    // dedicated .npy fixture file.  Tolerance rationale matches above:
    // ggml CPU has ~1e-3 relative FFN/gelu noise from the FP16 lookup,
    // amplified modestly by the QFormer FFN.
    for (const auto & bc : block_checks) {
        auto it = h.captured.find(bc.name);
        if (it == h.captured.end()) {
            std::fprintf(stdout, "  [MISS] %-26s : tensor never flowed through cb_eval\n", bc.name);
            failed++;
            continue;
        }
        g4v_npy::array_f32 expect;
        try {
            expect = g4v_npy::load_f32(fixtures_dir + "/" + bc.fixture);
        } catch (const std::exception & e) {
            std::fprintf(stdout, "  [FAIL] %-26s : %s\n", bc.name, e.what());
            failed++;
            continue;
        }
        const std::vector<float> & got = it->second;
        if (got.size() != static_cast<size_t>(expect.numel())) {
            std::fprintf(stdout, "  [SIZE] %-26s : got %zu floats, expected %lld\n",
                         bc.name, got.size(), (long long) expect.numel());
            failed++;
            continue;
        }
        auto rep = g4v_npy::diff_stats(got.data(), expect.data.data(), got.size());
        const double rel_to_ref_max = rep.ref_p99 > 0
            ? rep.max_abs / rep.ref_p99
            : rep.max_abs;
        const bool pass = rel_to_ref_max <= 5e-3 && rep.mean_abs <= 5e-3;
        std::fprintf(stdout,
                     "  [%s] %-26s : max_abs=%.4g  mean_abs=%.4g  max_abs/ref_max=%.4g  ref_max=%.4g\n",
                     pass ? " OK " : "FAIL", bc.name,
                     rep.max_abs, rep.mean_abs, rel_to_ref_max, rep.ref_p99);
        if (!pass) failed++;
    }

    clip_free(init.ctx_v);
    if (init.ctx_a) clip_free(init.ctx_a);
    g_harness = nullptr;

    if (failed > 0) {
        std::fprintf(stdout, "%d checkpoint(s) failed\n", failed);
        return 1;
    }
    std::fprintf(stdout, "all checkpoints passed\n");
    return 0;
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --mmproj <mmproj.gguf> --fixtures <fixtures_dir>\n"
        "\n"
        "  mmproj     path to the converted Granite Vision 4.1 mmproj GGUF\n"
        "  fixtures   directory containing siglip_input.npy and siglip_hidden_states.npy\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string mmproj;
    std::string fixtures;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) {
            mmproj = argv[++i];
        } else if (a == "--fixtures" && i + 1 < argc) {
            fixtures = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (mmproj.empty() || fixtures.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    try {
        return run(mmproj, fixtures);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 4;
    }
}
