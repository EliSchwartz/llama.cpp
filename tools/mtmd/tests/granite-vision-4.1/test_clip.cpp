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
    // We exercise a sparse set — enough to catch divergence early without
    // spamming stdout.  Fill in more as debugging demands it.
    std::vector<checkpoint> checkpoints = {
        {"pos_embed",    "siglip_hidden_states_layer_0.npy"},   // optional, see below
        {"layer_out-0",  "siglip_hidden_states_layer_1.npy"},
        {"layer_out-6",  "siglip_hidden_states_layer_7.npy"},
        {"layer_out-12", "siglip_hidden_states_layer_13.npy"},
        {"layer_out-18", "siglip_hidden_states_layer_19.npy"},
        {"layer_out-26", "siglip_hidden_states_layer_27.npy"},  // final layer
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
            std::fprintf(stdout, "  [MISS] %-18s : tensor never flowed through cb_eval\n", cp.name.c_str());
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
            std::fprintf(stdout, "  [SKIP] %-18s : no fixture mapping\n", cp.name.c_str());
            continue;
        }
        const float * expected = hs.data.data() + layer * layer_stride;
        const std::vector<float> & got = it->second;
        if (static_cast<int64_t>(got.size()) != layer_stride) {
            std::fprintf(stdout, "  [SIZE] %-18s : got %zu floats, expected %lld\n",
                         cp.name.c_str(), got.size(), static_cast<long long>(layer_stride));
            failed++;
            continue;
        }
        auto [mx, mn] = g4v_npy::diff_stats(got.data(), expected, static_cast<size_t>(layer_stride));
        const bool pass = mx <= 1e-3;
        std::fprintf(stdout, "  [%s] %-18s : max_abs=%.6g  mean_abs=%.6g\n",
                     pass ? " OK " : "FAIL", cp.name.c_str(), mx, mn);
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
