// UCVG — Universal Control Vector Generator


#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "ucvg-util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#ifdef _WIN32
# include <direct.h>
# define UCVG_MKDIR(p) _mkdir(p)
#else
# include <sys/stat.h>
# include <sys/types.h>
# define UCVG_MKDIR(p) mkdir((p), 0755)
#endif

namespace {

[[maybe_unused]] constexpr uint32_t UCVG_MAGIC = 0x55435647u;
[[maybe_unused]] constexpr uint32_t UCVG_VERSION = 1u;
constexpr float EPS = 1e-12f;

// Print paths with forward slashes (the -o arg and OS-joined parts mix `\` and `/`).
static std::string to_slash(std::string s) { for (auto & c : s) if (c == '\\') c = '/'; return s; }

// ---- Unified pipeline timer ----
struct UcvgsTimerStage { std::string name; double start = 0.0, end = 0.0, elapsed = 0.0; int level = 0, parent = -1; };
class UcvgsUnifiedTimer {
public:
    std::vector<UcvgsTimerStage> stages;
    std::vector<int> stack;
    double pipeline_start = 0.0, pipeline_end = 0.0;
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void start_pipeline() { pipeline_start = now(); }
    void end_pipeline()   { pipeline_end = now(); }
    int begin(const std::string & name) {
        const int parent = stack.empty() ? -1 : stack.back();
        const int level  = (int)stack.size();
        const int idx = (int)stages.size();
        stages.push_back({ name, now(), 0.0, 0.0, level, parent });
        stack.push_back(idx); return idx;
    }
    void finish(int idx) {
        if (stack.empty() || stack.back() != idx) return;   // safety: only close the matching open stage
        const double e = now(); stages[idx].end = e; stages[idx].elapsed = e - stages[idx].start; stack.pop_back();
    }
    static std::string fmt(double s) { char b[32]; if (s < 60.0) snprintf(b, sizeof(b), "%.1fs", s); else { int m = (int)(s / 60.0); snprintf(b, sizeof(b), "%dm %.0fs", m, s - m * 60.0); } return b; }
    void print_summary() const {
        // Clean indented tree: 'total' = actual wall-clock (not a stage sum); hierarchy shown by indentation.
        const double actual = (pipeline_start > 0.0 && pipeline_end > 0.0) ? (pipeline_end - pipeline_start) : 0.0;
        std::printf("\n%-52s %10s\n", "total", fmt(actual).c_str());
        for (const auto & s : stages) {
            if (s.name == "Save report" || s.name == "Compute layer diagnostics") continue;   // dropped on request
            const std::string label = std::string((size_t)(s.level + 1) * 2, ' ') + s.name;
            std::printf("%-52s %10s\n", label.c_str(), fmt(s.elapsed).c_str());
        }
    }
};
static UcvgsUnifiedTimer g_timer;
static bool g_output_model_responses = false;   // --output-model-responses: dump every model response (full raw text) at each generation site

// Device / memory settings, applied at every model-load + context-creation site.
static int32_t  g_ngl          = -1;              // -ngl / --gpu-layers; -1 = all layers to GPU (llama.cpp default)
static ggml_type g_cache_type_k = GGML_TYPE_Q8_0; // -ctk / --cache-type-k
static ggml_type g_cache_type_v = GGML_TYPE_Q8_0; // -ctv / --cache-type-v

struct UcvgsStageScope { int idx = -1; explicit UcvgsStageScope(const char * n) : idx(g_timer.begin(n)) {} ~UcvgsStageScope() { g_timer.finish(idx); } };

struct Params {
    std::string in;
    std::string out;
    int model_layer_count = 0; // 0 -> use layer count from the activation header
    float threshold_fraction = 0.0f;
    int n_bootstrap = 100;
    float subsample_frac = 0.4f;
    float consistency_threshold = 0.0f;
    // float consistency_threshold = 1.0f;
    bool apply_global_scaling = true;
    float snr_smoothing_sigma = 1.0f;
    float snr_weight_power = 1.0f;
    bool depth_envelope_enabled = false;
    float depth_envelope_center_frac = 0.46f;
    float depth_envelope_width_frac = 0.62f;
    float depth_envelope_sharpness = 0.05f;
    float snr_blend = 0.5f;
    uint64_t seed = 0; // 0 -> random_device
    bool use_pca = false; // --pca: build PC1 via power iteration and use it as the steering direction; report cosine vs mean-diff
    int capture_slots = 1;  // --capture-slots: N parallel contexts for stage-2/3 activation capture (1 = serial, current)
    int capture_ctx   = 512; // --capture-ctx: per-slot n_ctx for capture (prefill-only; must fit the worst-case prompt)
};

// Write direction.{1..n_layers} tensors (F32, zero for unselected) + the per-layer center scalars as a single metadata
// array "controlvector.center" (one F32 each) to a .gguf with arch "controlvector".
// center[l] is (mu_l . v_l): the default-state projection onto that layer's direction, consumed by the multiplicative
// (gain) steering path. Storing it in metadata (not tensors) keeps the file loadable by stock llama.cpp, whose CV loader
// reads only direction.* and ignores unknown KV keys.
static bool export_gguf(const std::string & fname, int n_layers, int d, const std::vector<std::vector<float>> & layers,
                        const std::vector<float> & center) {
    struct ggml_init_params params_ggml = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(std::max(1, n_layers) + 2),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * tctx = ggml_init(params_ggml);
    if (!tctx) return false;

    std::vector<struct ggml_tensor *> tensors((size_t)n_layers, nullptr);
    for (int l = 0; l < n_layers; ++l) {
        struct ggml_tensor * t = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, (int64_t)d);
        if (!t) return false;
        t->data = malloc(ggml_nbytes(t));
        std::vector<float> v = l < (int)layers.size() ? layers[l] : std::vector<float>((size_t)d, 0.0f);
        v.resize((size_t)d, 0.0f);
        memcpy(t->data, v.data(), (size_t)d * sizeof(float));
        char name[64];
        snprintf(name, sizeof(name), "direction.%d", l + 1); // 1-based, matching the Python writer
        ggml_set_name(t, name);
        tensors[l] = t;
    }

    struct gguf_context * gctx = gguf_init_empty();
    gguf_set_val_str(gctx, "general.architecture", "controlvector");
    gguf_set_val_i32(gctx, "controlvector.layer_count", n_layers);
    // center scalars (c_l = mu_l . v_l) stored as a single F32 array in metadata, not as tensors: the stock CV
    // loader reads only direction.* tensors and ignores unknown KV keys, so this stays stock-compatible.
    if (!center.empty()) {
        gguf_set_arr_data(gctx, "controlvector.center", GGUF_TYPE_FLOAT32, center.data(), center.size());
    }
    for (auto * t : tensors) gguf_add_tensor(gctx, t);

    bool ok = gguf_write_to_file(gctx, fname.c_str(), false);
    gguf_free(gctx);
    return ok;
}

// PC1 via power iteration (Stanford CS168 lecture 8 "PCA and the Power Iteration Method";
// arXiv 2410.23999 "A Power Method for Computing Singular Value Decomposition")
static bool run(const Params & p, uint32_t N, uint32_t L, uint32_t d,
                const std::vector<float> & pos, const std::vector<float> & neg) {
    const int nL = (int)L;
    const int nd = (int)d;

    std::printf("    Building CV: N=%u layers=%u d=%u\n", N, L, d);

    // --- Step 1: global means + variance denominators ---
    std::vector<std::vector<float>> mu((size_t)nL, std::vector<float>((size_t)nd, 0.0f));
    std::vector<float> denom((size_t)nL, 0.0f);
    for (int l = 0; l < nL; ++l) {
        std::vector<float> m((size_t)nd, 0.0f);
        for (uint32_t n = 0; n < N; ++n) {
            const float * pp = &pos[((size_t)n * nL + l) * nd];
            const float * nn = &neg[((size_t)n * nL + l) * nd];
            for (int e = 0; e < nd; ++e) { m[e] += pp[e]; m[e] += nn[e]; }
        }
        for (int e = 0; e < nd; ++e) m[e] /= (float)(2 * (int)N);
        for (int e = 0; e < nd; ++e) mu[(size_t)l][e] = m[e];
        // denominator
        float s = 0.0f;
        for (uint32_t n = 0; n < N; ++n) {
            const float * pp = &pos[((size_t)n * nL + l) * nd];
            const float * nn = &neg[((size_t)n * nL + l) * nd];
            float np_ = 0.0f, nn_ = 0.0f;
            for (int e = 0; e < nd; ++e) { float a = pp[e] - m[e]; float b = nn[e] - m[e]; np_ += a * a; nn_ += b * b; }
            s += std::sqrt(np_) + std::sqrt(nn_);
        }
        denom[l] = s / (float)(2 * (int)N);
    }

    // --- Step 3: bootstrap sign-stability masking ---
    std::vector<float> raw_snr((size_t)nL, 0.0f);
    std::vector<float> masked_snr((size_t)nL, 0.0f);
    std::vector<float> kept_frac((size_t)nL, 0.0f);
    std::vector<std::vector<float>> raw_delta_norm((size_t)nL, { 0.0f }); // per-layer ||delta_full||
    std::vector<std::vector<float>> sparse_dir((size_t)nL, std::vector<float>((size_t)nd, 0.0f));

    const int k = std::max(2, (int)((float)N * p.subsample_frac));
    std::mt19937_64 rng(p.seed != 0 ? p.seed : (uint64_t)std::random_device{}());
    std::vector<uint32_t> perm((size_t)N);
    for (uint32_t i = 0; i < N; ++i) perm[i] = i;

    for (int l = 0; l < nL; ++l) {
        // delta_full from RAW means (uncentered)
        std::vector<float> delta((size_t)nd, 0.0f);
        for (uint32_t n = 0; n < N; ++n) {
            const float * pp = &pos[((size_t)n * nL + l) * nd];
            const float * nn = &neg[((size_t)n * nL + l) * nd];
            for (int e = 0; e < nd; ++e) delta[e] += pp[e] - nn[e];
        }
        for (int e = 0; e < nd; ++e) delta[e] /= (float)(int)N;
        float dn = 0.0f; for (int e = 0; e < nd; ++e) dn += delta[e] * delta[e]; dn = std::sqrt(dn);
        raw_delta_norm[l][0] = dn;
        raw_snr[l] = dn / (denom[l] + EPS);

        // bootstrap over CENTERED vectors (pos - mu, neg - mu)
        const float * mul = &mu[(size_t)l][0];
        std::vector<std::vector<float>> B((size_t)p.n_bootstrap, std::vector<float>((size_t)nd, 0.0f));
        for (int b = 0; b < p.n_bootstrap; ++b) {
            // partial Fisher-Yates: draw k distinct indices
            for (uint32_t i = 0; i < N; ++i) perm[i] = i;
            for (int i = 0; i < k; ++i) {
                const uint64_t r = rng();
                const uint32_t j = (uint32_t)(i + (r % (uint64_t)(N - i)));
                std::swap(perm[i], perm[j]);
            }
            std::vector<float> wpos((size_t)nd, 0.0f), wneg((size_t)nd, 0.0f);
            for (int t = 0; t < k; ++t) {
                const uint32_t n = perm[t];
                const float * pp = &pos[((size_t)n * nL + l) * nd];
                const float * nn = &neg[((size_t)n * nL + l) * nd];
                for (int e = 0; e < nd; ++e) { wpos[e] += pp[e] - mul[e]; wneg[e] += nn[e] - mul[e]; }
            }
            for (int e = 0; e < nd; ++e) B[(size_t)b][e] = wpos[e] / k - wneg[e] / k;
        }

        // mean_dir, sign consistency, mask
        std::vector<float> mean_dir((size_t)nd, 0.0f);
        for (int e = 0; e < nd; ++e) { float s = 0.0f; for (int b = 0; b < p.n_bootstrap; ++b) s += B[(size_t)b][e]; mean_dir[e] = s / p.n_bootstrap; }
        int kept = 0;
        for (int e = 0; e < nd; ++e) {
            float sc = 0.0f; const float ms = sgnf(mean_dir[e]);
            for (int b = 0; b < p.n_bootstrap; ++b) if (sgnf(B[(size_t)b][e]) == ms) sc += 1.0f;
            sc /= (float)p.n_bootstrap;
            if (sc >= p.consistency_threshold - 1e-9f) { sparse_dir[l][e] = mean_dir[e]; ++kept; } else { sparse_dir[l][e] = 0.0f; }
        }
        kept_frac[l] = (float)kept / (float)nd;
        float mn = 0.0f; for (int e = 0; e < nd; ++e) mn += sparse_dir[l][e] * sparse_dir[l][e];
        masked_snr[l] = std::sqrt(mn) / (denom[l] + EPS);
    }

    // --- PC1 (power iteration) + EVR + cosine vs raw mean-diff - only when --pca;
    std::vector<std::vector<float>> pc1_dir((size_t)nL, std::vector<float>((size_t)nd, 0.0f));
    std::vector<float> pca_cosine((size_t)nL, 0.0f), pca_evr((size_t)nL, 0.0f);
    std::vector<bool>  pca_ok((size_t)nL, false);
    if (p.use_pca) {
        const int m = (int)(2 * N);   // pooled rows for the DIRECTION: N positive + N negative
        std::vector<float> Xc((size_t)m * nd);   // centered combined matrix -> PC1 direction
        std::vector<float> Dc((size_t)N * nd);   // centered difference vectors (pos-neg, mean-subtracted) -> EVR metric
        for (int l = 0; l < nL; ++l) {
            // raw mean-diff direction (== delta_full in Step 3): mean_n(pos[n] - neg[n])
            std::vector<float> raw((size_t)nd, 0.0f);
            for (uint32_t n = 0; n < N; ++n) { const float * pp = &pos[((size_t)n * nL + l) * nd]; const float * nn = &neg[((size_t)n * nL + l) * nd]; for (int e = 0; e < nd; ++e) raw[e] += pp[e] - nn[e]; }
            for (int e = 0; e < nd; ++e) raw[e] /= (float)N;
            float nr = 0.0f; for (int e = 0; e < nd; ++e) nr += raw[e] * raw[e]; nr = std::sqrt(nr);
            // centered combined matrix Xc: rows 0..N-1 = pos - mu[l], rows N..2N-1 = neg - mu[l]   (for DIRECTION)
            const float * mul = &mu[(size_t)l][0];
            for (uint32_t n = 0; n < N; ++n) { const float * pp = &pos[((size_t)n * nL + l) * nd]; const float * nn = &neg[((size_t)n * nL + l) * nd]; for (int e = 0; e < nd; ++e) { Xc[(size_t)n * nd + e] = pp[e] - mul[e]; Xc[(size_t)(n + N) * nd + e] = nn[e] - mul[e]; } }
            // centered difference vectors Dc[i] = (pos_i - neg_i) - raw   (for EVR METRIC; mean(pos-neg) == raw); trD = total variance
            float trD = 0.0f;
            for (uint32_t n = 0; n < N; ++n) { const float * pp = &pos[((size_t)n * nL + l) * nd]; const float * nn = &neg[((size_t)n * nL + l) * nd]; for (int e = 0; e < nd; ++e) { float dv = pp[e] - nn[e] - raw[e]; Dc[(size_t)n * nd + e] = dv; trD += dv * dv; } }
            std::vector<float> pc1;
            if (nr > 1e-6f && compute_pc1(Xc.data(), m, nd, pc1, 60, 1e-6)) {
                float dot = 0.0f; for (int e = 0; e < nd; ++e) dot += pc1[e] * raw[e];
                float cosv = dot / (nr + EPS);   // pc1 is a unit vector
                if (cosv < 0.0f) { for (int e = 0; e < nd; ++e) pc1[e] = -pc1[e]; cosv = -cosv; }   // align PC1 toward mean-diff
                pca_cosine[l] = cosv; pca_ok[l] = true; pc1_dir[(size_t)l] = std::move(pc1);   // DIRECTION = pooled PC1 (unchanged)
                // EVR METRIC = top eigenvalue ratio of the centered difference matrix: lam1 / trace
                std::vector<float> pc1d;
                if (trD > 1e-9f && compute_pc1(Dc.data(), (int)N, nd, pc1d, 60, 1e-6)) {
                    float lam1d = 0.0f; for (uint32_t n = 0; n < N; ++n) { float acc = 0.0f; const float * xr = &Dc[(size_t)n * nd]; for (int e = 0; e < nd; ++e) acc += xr[e] * pc1d[e]; lam1d += acc * acc; }   // ||Dc*pc1d||^2
                    pca_evr[l] = lam1d / trD;
                } else {
                    pca_evr[l] = 0.0f;   // degenerate difference matrix (no signal)
                }
            } else {
                pca_ok[l] = false; pca_evr[l] = 0.0f;   // degenerate direction: fall back to the mean-diff direction below
            }
        }
    }

    // --- Step 4: metric smoothing -- masked_snr (mean-diff) by default; EVR of centered difference vectors when --pca ---
    const std::vector<float> & metric = p.use_pca ? pca_evr : masked_snr;
    std::vector<float> smoothed = smooth_curve(metric, p.snr_smoothing_sigma);

    // --- Step 5: depth envelope ---
    std::vector<float> envelope((size_t)nL, 1.0f);
    std::vector<float> effective((size_t)nL, 0.0f);
    if (p.depth_envelope_enabled) {
        envelope = compute_depth_envelope(nL, p.depth_envelope_center_frac, p.depth_envelope_width_frac, p.depth_envelope_sharpness);
        for (int l = 0; l < nL; ++l) effective[l] = smoothed[l] * envelope[l];
    } else {
        effective = smoothed;
    }

    // --- Step 6: layer selection & weighting ---
    float max_smoothed = *std::max_element(smoothed.begin(), smoothed.end());
    float max_effective = *std::max_element(effective.begin(), effective.end());
    const float threshold = p.threshold_fraction * max_effective;
    std::vector<int> selected;
    for (int l = 0; l < nL; ++l) if (effective[l] >= threshold) selected.push_back(l);
    if (selected.empty()) { int best = 0; for (int l = 1; l < nL; ++l) if (effective[l] > effective[best]) best = l; selected.push_back(best); }

    std::vector<float> layer_w((size_t)nL, 0.0f);
    std::vector<std::vector<float>> unscaled((size_t)nL, std::vector<float>((size_t)nd, 0.0f));
    for (int l : selected) {
        const float ratio = effective[l] / (max_effective + EPS);
        const float metric_weight = std::pow(ratio, p.snr_weight_power);
        const float envelope_weight = envelope[l];
        const float w_l = p.snr_blend * metric_weight + (1.0f - p.snr_blend) * envelope_weight;
        layer_w[l] = w_l;
        for (int e = 0; e < nd; ++e) unscaled[(size_t)l][e] = w_l * (p.use_pca && pca_ok[l] ? pc1_dir[(size_t)l][e] : sparse_dir[(size_t)l][e]);   // --pca: PC1 direction, else mean-diff (unchanged)
    }

    // --- Step 7: global alpha ---
    // TODO global alpha scaling stopped working after some changes so scales are back at +-0.1 instead of the intended +-1
    // global scaling does make scales between different models/traits more understandable but doesnt actually scale to +-1
    // also that 0.75 coefficient means nothing at all                  vvvvv
    float P_baseline = 0.0f; for (int l = 0; l < nL; ++l) P_baseline += 0.75f * raw_delta_norm[l][0];       
    float P_current = 0.0f;
    for (int l : selected) { float s = 0.0f; for (int e = 0; e < nd; ++e) s += unscaled[(size_t)l][e] * unscaled[(size_t)l][e]; P_current += std::sqrt(s); }
    const float alpha = P_current > 0.0f ? P_baseline / P_current : 1.0f;

    // --- Step 8: final vectors + export ---
    const int n_export = p.model_layer_count > 0 ? p.model_layer_count : nL;
    std::vector<std::vector<float>> layers((size_t)n_export, std::vector<float>((size_t)nd, 0.0f));
    for (int l : selected) if (l < n_export) { const float sc = p.apply_global_scaling ? alpha : 1.0f; for (int e = 0; e < nd; ++e) layers[(size_t)l][e] = unscaled[(size_t)l][e] * sc; }

    // diagnostics: only the selected-layer summary is shown (Max Smoothed/Effective/Threshold + Global Factor dropped on request).
    (void)max_smoothed;   // no longer printed; still computed above for the selection/weight math
    std::printf("    Selected Layers: %zu / %d -> [", selected.size(), nL);
    for (size_t i = 0; i < selected.size(); ++i) std::printf("%s%d", i ? ", " : "", selected[i]);
    std::printf("]\n");
    // Per-layer detail line (raw_snr / masked_snr / smth / eff / kept / w) removed on request.
    // The all-layers mean-diff SNR / EVR / PC1-cos graph below is now the per-layer summary.
    /* for (int l = 0; l < nL; ++l) {
        if (std::find(selected.begin(), selected.end(), l) == selected.end()) continue;
        std::printf("  layer %3d | raw_snr %.4f masked_snr %.4f smth %.4f eff %.4f kept %.4f w %.4f\n",
                    l, raw_snr[l], masked_snr[l], smoothed[l], effective[l], kept_frac[l], layer_w[l]);
    } */
    (void)raw_snr; (void)kept_frac; (void)layer_w;   // still computed for the math above; detail line no longer printed

    // Per-layer graph: mean-diff SNR + final weight always shown; EVR(centered diffs) + PC1 cos only under --pca.
    {
        const float max_snr = *std::max_element(masked_snr.begin(), masked_snr.end());
        const float max_w   = *std::max_element(layer_w.begin(), layer_w.end());
        const float max_evr = p.use_pca ? *std::max_element(pca_evr.begin(), pca_evr.end()) : 0.0f;
        auto mbar = [](float v, float vmax, int bw) {
            int k = 0; if (vmax > 1e-9f) { double f = (double)v / vmax; if (f < 0) f = 0; if (f > 1) f = 1; k = (int)(f * bw + 0.5); }
            std::string b; for (int i = 0; i < bw; ++i) b += (i < k ? "#" : "."); return b;
        };
        const int bw = 12;
        std::printf("\n    %s   (all %d layers; * = selected)\n",
                    p.use_pca ? "mean-diff SNR / EVR(centered diffs) / PC1 cos / final weight" : "mean-diff SNR / final weight", nL);
        double csum = 0.0; int cnt = 0;
        for (int l2 = 0; l2 < nL; ++l2) {
            const bool sel = std::find(selected.begin(), selected.end(), l2) != selected.end();
            if (p.use_pca) {
                if (!pca_ok[l2]) { std::printf("      layer %3d | DEGENERATE (no PC1)\n", l2); continue; }
                csum += pca_cosine[l2]; ++cnt;
                std::printf("      layer %3d | snr %7.4f [%s]  evr %7.4f [%s]  cos %7.4f  w %7.4f [%s]%s\n",
                            l2, (double)masked_snr[l2], mbar(masked_snr[l2], max_snr, bw).c_str(),
                            (double)pca_evr[l2],       mbar(pca_evr[l2], max_evr, bw).c_str(),
                            (double)pca_cosine[l2],    (double)layer_w[l2], mbar(layer_w[l2], max_w, bw).c_str(), sel ? " *" : "");
            } else {
                std::printf("      layer %3d | snr %7.4f [%s]  w %7.4f [%s]%s\n",
                            l2, (double)masked_snr[l2], mbar(masked_snr[l2], max_snr, bw).c_str(),
                            (double)layer_w[l2], mbar(layer_w[l2], max_w, bw).c_str(), sel ? " *" : "");
            }
        }
        if (p.use_pca) std::printf("      average cosine across all layers: %.4f  (%d/%d valid)\n", cnt ? csum / (double)cnt : 0.0, cnt, nL);
    }

    // center scalar per layer: c_l = mu_l . v_l  (mu = Step-1 centering, v_l = final direction)
    std::vector<float> center((size_t)n_export, 0.0f);
    for (int l = 0; l < nL && l < n_export; ++l) {
        float dot = 0.0f;
        for (int e = 0; e < nd; ++e) dot += mu[(size_t)l][e] * layers[(size_t)l][e];
        center[(size_t)l] = dot;
    }
    const bool ok = export_gguf(p.out, n_export, nd, layers, center);
    if (ok) std::printf("    Wrote the CV to %s\n    CV generation is done, everything after this point is for probing the recommended usage amplitude and evaluating the results", to_slash(p.out).c_str());
    return ok;
}


// Read a text file as a list of non-empty lines (trailing \r stripped).
static std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// Capture the last-token-of-prefill hidden state per layer for one (system, user) pair. With made up special tokens it looks somewhat like this:
// <system_prompt_start>...<system_prompt_end><user_turn_start>...<user_turn_end><assistant_turn_start><reasoning_start><reasoning_end> and at this point we read the activations. 
static std::vector<float> capture_pair(const llama_model * model, llama_context * ctx,
                                       const common_chat_templates * tmpls,
                                       const std::string & system, const std::string & user,
                                       uint32_t L, int n_embd) {
    common_chat_msg sys; sys.role = "system"; sys.content = system;
    common_chat_msg usr; usr.role = "user";   usr.content = user;
    std::vector<common_chat_msg> msgs;
    msgs.push_back(sys); msgs.push_back(usr);

    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.messages = msgs;
    in.add_generation_prompt = true;
    in.enable_thinking = false;
    auto res = common_chat_templates_apply(tmpls, in);
    const std::string prompt = res.prompt;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 64);
    int ntok = llama_tokenize(vocab, prompt.data(), (int)prompt.size(), toks.data(), (int)toks.size(), true /*add_special*/, true /*parse_special*/);
    if (ntok < 0) return {};
    toks.resize((size_t)ntok);

    std::vector<int32_t> layers(L);
    for (uint32_t i = 0; i < L; ++i) layers[i] = (int32_t)i;
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_capture_layers(ctx, layers.data(), (int)L);
    llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
    if (llama_decode(ctx, batch) != 0) return {};

    std::vector<float> acts((size_t)L * n_embd);
    llama_get_captured_activations(ctx, acts.data());
    return acts;
}

// Read the trained scenarios from a stimuli.json
static std::vector<std::string> read_trained_scenarios(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::vector<std::string> out;
    std::string line;
    auto unesc = [](const std::string & s) {
        std::string o; o.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '\\' && i + 1 < s.size()) { char n = s[++i]; o += (n == 'n') ? '\n' : (n == 'r') ? '\r' : (n == 't') ? '\t' : n; }
            else o += c;
        }
        return o;
    };
    while (std::getline(f, line)) {
        size_t k = line.find("\"scenario\"");
        if (k == std::string::npos) continue;
        size_t i = line.find(':', k);
        if (i == std::string::npos) continue;
        ++i;
        while (i < line.size() && line[i] != '"') ++i;
        if (i >= line.size()) continue;
        ++i; // past the opening quote of the value
        std::string raw;
        while (i < line.size()) {
            char c = line[i];
            if (c == '\\' && i + 1 < line.size()) { raw += c; ++i; raw += line[i]; }
            else if (c == '"') break;
            else raw += c;
            ++i;
        }
        std::string scenario = unesc(raw);
        bool trained = true;
        size_t tk = line.find("\"trained_on\"");
        if (tk != std::string::npos) {
            size_t j = line.find(':', tk);
            if (j != std::string::npos) { ++j; while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) ++j; trained = (line.compare(j, 4, "true") == 0); }
        }
        if (trained && !scenario.empty()) out.push_back(scenario);
    }
    return out;
}


// ---- Stage 1: scenario generation ----

static std::string json_escape(const std::string & s) {
    std::string o; o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// System prompt for scenario generation.
static const char * kScenarioGenPrompt = R"UCVG(You are an expert synthetic data generator for representation engineering and mechanistic interpretability research.
Your goal is to output a numbered list of scenarios that would elicit contrasting behaviors for the two described traits.

The scenarios must be neutral situations where the difference between the two traits would become noticeable. They must not lean toward one persona or the other, and must not pre-solve how the person reacts.
Describe only the situation or event itself; do not narrate what the person does, feels, thinks, or decides in response.

Make the scenarios diverse across different real-world contexts (social, professional, domestic, technical, travel, health, work, etc.) and avoid near-duplicates or repeated templates. Each must be a concrete, plausible everyday situation.

Do not use trait names, category labels, or self-diagnostic words inside the scenarios.

Scenarios must be in different languages:
20% in English
20% in Chinese
20% in Italian
20% in Russian
20% in Japanese

Output MUST be a plain text numbered list (1., 2., 3., ...) with one scenario per line, each formatted as "N. <scenario>". No JSON, no markdown, no code fences, no comments, no blank lines. Output exactly N lines for N requested scenarios.
)UCVG";

// Sample the next token from softmax(logits / temperature).
static int sample_softmax(const float * logits, int n_vocab, float temp, std::mt19937 & rng) {
    if (!std::isfinite(temp) || temp <= 0.0f) temp = 1e-4f;
    const float mx = *std::max_element(logits, logits + n_vocab);
    double sum = 0.0;
    for (int t = 0; t < n_vocab; ++t) sum += std::exp((double)(logits[t] - mx) / temp);
    if (!std::isfinite(sum) || sum <= 0.0) { int bi = 0; for (int t = 1; t < n_vocab; ++t) if (logits[t] > logits[bi]) bi = t; return bi; }
    std::uniform_real_distribution<double> dist(0.0, sum);
    const double u = dist(rng);
    double acc = 0.0;
    for (int t = 0; t < n_vocab; ++t) { acc += std::exp((double)(logits[t] - mx) / temp); if (u <= acc) return t; }
    return n_vocab - 1;
}

// Template a (system, user) chat, then decode-and-sample until an end-of-generation token (is_eog: eos/eot/eom) or max_tokens.
static std::string generate_text(const llama_model * model, llama_context * ctx,
                                 const common_chat_templates * tmpls,
                                 const std::string & system, const std::string & user,
                                 int max_tokens, float temperature, std::mt19937 & rng, const char * dbg_label, bool quiet = false, int * n_tokens_out = nullptr, double * seconds_out = nullptr, int * prompt_tokens_out = nullptr) {
    common_chat_msg sys; sys.role = "system"; sys.content = system;
    common_chat_msg usr; usr.role = "user";   usr.content = user;
    std::vector<common_chat_msg> msgs;
    msgs.push_back(sys); msgs.push_back(usr);

    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.messages = msgs;
    in.add_generation_prompt = true;
    in.enable_thinking = false;
    auto res = common_chat_templates_apply(tmpls, in);
    const std::string prompt = res.prompt;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 64);
    int ntok = llama_tokenize(vocab, prompt.data(), (int)prompt.size(), toks.data(), (int)toks.size(), true /*add_special*/, true /*parse_special*/);
    if (ntok < 0) return {};
    toks.resize((size_t)ntok);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_memory_clear(llama_get_memory(ctx), true); // fresh context for each generation
    const size_t n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < toks.size(); ) {   // chunk the prefill so a long prompt never exceeds a single n_batch
        const size_t chunk = std::min(n_batch, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, chunk)) != 0) return {};
        i += chunk;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string text; text.reserve(4096);
    int n_gen = 0;
    for (int n = 0; n < max_tokens; ++n) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        const int next = sample_softmax(logits, n_vocab, temperature, rng);
        if (llama_vocab_is_eog(vocab, (llama_token)next)) break;   // stop on any end-of-generation token (eos/eot/eom) — same predicate as llama-server
        std::string piece = common_token_to_piece(vocab, (llama_token)next, true);
        text += piece;
        ++n_gen;
        llama_token t = (llama_token)next;
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) break;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!quiet && n_gen > 0 && sec > 0.0) std::printf("      generated %d tokens in %.1fs (%.1f tok/s)\n", n_gen, sec, (double)n_gen / sec);
    if (n_tokens_out) *n_tokens_out = n_gen;
    if (seconds_out) *seconds_out = sec;
    if (prompt_tokens_out) *prompt_tokens_out = ntok;
    // --output-model-responses: print the complete raw response at every generation site.
    if (g_output_model_responses) { std::fprintf(stdout, "\n[%s - %d generated tokens] %s\n", dbg_label, n_gen, text.c_str()); std::fflush(stdout); }
    return text;
}

// Parse a numbered list ("1. text" or "1) text"), stripping code fences and blank lines

// One persona-pair slot: generate a numbered list of exactly `want` scenarios
// The model may under- or over-fill the list; we take the first `want` when it over-produces and,
// across retries, keep the attempt whose count is closest to the target
static std::vector<std::string> gen_slot(const llama_model * model, llama_context * ctx, const common_chat_templates * tmpls,
        const std::string & user_prompt, int want, int max_attempts, std::mt19937 & rng) {
    const int maxtok = std::min(3840, std::max(512, 50 * want));   // ~50 tokens per scenario (bounded), no runaway
    if (getenv("UCVG_DEBUG_PROMPT")) {
        std::fprintf(stderr, "\n===== UCVG DEBUG: SYSTEM PROMPT (%zu bytes) =====\n%s\n===== UCVG DEBUG: USER PROMPT =====\n%s\n=====\n",
                     strlen(kScenarioGenPrompt), kScenarioGenPrompt, user_prompt.c_str());
    }
    std::vector<std::string> best;   // the attempt closest to the target (largest count, capped at `want`)
    for (int attempt = 1; attempt <= max_attempts && (int)best.size() < want; ++attempt) {
        const std::string raw = generate_text(model, ctx, tmpls, kScenarioGenPrompt, user_prompt, maxtok, 1.0f, rng, "stage1/scenario-generation");
        if (getenv("UCVG_DEBUG_PROMPT")) std::fprintf(stderr, "\n===== UCVG DEBUG: RAW LLM OUTPUT (%zu chars) =====\n%s\n=====\n", raw.size(), raw.c_str());
        std::vector<std::string> got = parse_numbered(raw);
        if ((int)got.size() > want) got.resize(want);   // take the first `want` (guarantee)
        std::printf("      attempt %d: parsed %d (wanted: %d)\n", attempt, (int)got.size(), want);
        if ((int)got.size() > (int)best.size()) best = got;   // keep the one closest to target (not just the last)
    }
    if ((int)best.size() != want) std::printf("  note: slot yielded %d scenarios (wanted %d)\n", (int)best.size(), want);
    return best;
}


// ---- Resumable pipeline: ----

static bool file_exists(const std::string & p) { std::ifstream f(p); return (bool)f; }

static std::string slugify(const std::string & s) {
    std::string o;
    for (char c : s) o += (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) ? c : '_';
    while (!o.empty() && o.front() == '_') o.erase(o.begin());
    return o.empty() ? std::string("trait") : o;
}

// Write a stimuli.json (list of {"scenario","trained_on"}).
static bool write_stimuli_json(const std::string & path, const std::vector<std::pair<std::string, bool>> & stimuli) {
    std::ofstream jf(path);
    if (!jf) { std::fprintf(stderr, "ucvg: cannot write %s\n", path.c_str()); return false; }
    jf << "[\n";
    for (size_t i = 0; i < stimuli.size(); ++i) {
        jf << "  {\"scenario\": \"" << json_escape(stimuli[i].first) << "\", \"trained_on\": "
           << (stimuli[i].second ? "true" : "false") << "}";
        if (i + 1 < stimuli.size()) jf << ",";
        jf << "\n";
    }
    jf << "]\n";
    std::printf("    Wrote stimuli to %s\n", to_slash(path).c_str());
    return true;
}

// Stage 1 core: generate stimuli for all slots (caller loads model/ctx/tmpls).
static std::vector<std::pair<std::string, bool>> make_stimuli_core(const llama_model * model, llama_context * ctx,
        const common_chat_templates * tmpls,
        const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
        const std::string & scenario_desc, int pairs_per_slot, float reserve, int max_attempts, std::mt19937 & rng) {
    const int slots = (int)pos_facets.size();
    std::vector<std::pair<std::string, bool>> stimuli;
    for (int i = 0; i < slots; ++i) {
        const std::string user_prompt =
            "Recommended scenario direction: " + scenario_desc + "\n\n"
            "Positive persona: " + pos_facets[i] + "\n\n"
            "Negative persona: " + neg_facets[i] + "\n\n"
            "Now produce a numbered list with exactly " + std::to_string(pairs_per_slot) + " scenarios.\n"
            "Each line must start with the number (1., 2., 3., ...) followed by the scenario text.\n"
            "Output ONLY the numbered list, one scenario per line. No JSON, no markdown, no extra text.";
        std::printf("    slot %d/%d: generating...\n", i + 1, slots);
        const std::vector<std::string> got = gen_slot(model, ctx, tmpls, user_prompt, pairs_per_slot, max_attempts, rng);
        if (got.empty()) std::printf("slot %d/%d: discarding after %d attempt(s)\n", i + 1, slots, max_attempts);
        for (auto & sc : got) stimuli.emplace_back(sc, false);
    }

    const int total = (int)stimuli.size();
    const int eval_target = std::clamp((int)((double)reserve * (double)total + 0.5), 0, total);   // e.g. 0.15*60 -> 9
    for (auto & st : stimuli) st.second = true;                       // default: training
    std::vector<int> idx(total); for (int i = 0; i < total; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int e = 0; e < eval_target; ++e) stimuli[(size_t)idx[e]].second = false;   // exactly these are held out
    const int train_count = total - eval_target;
    std::printf("    generated %zu scenarios (%d for training, %zu for evaluation)\n", stimuli.size(), train_count, stimuli.size() - (size_t)train_count);
    return stimuli;
}

// Capture all pairs into pos/neg ([n][l][e], n = s*nF+f row-major).
// capture_slots == 1 -> one context, serial (current behavior). > 1 -> that many contexts from the SHARED model
// (read-only weights), a worker pool over disjoint pair indices. Safe: all per-generation state (KV, compute
// scratch, capture_layers/captured_acts, graph cache) is per-context; only read-only model/vocab/templates are
// shared; results write to disjoint pair regions. Mirrors how the in-tree server runs concurrent slots.
static bool capture_all(llama_model * model, const common_chat_templates * tmpls,
                        const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
                        const std::vector<std::string> & scenarios, int n_embd, uint32_t L,
                        int capture_ctx, int capture_slots, std::vector<float> & pos, std::vector<float> & neg) {
    const int nF = (int)pos_facets.size();
    if (nF <= 0 || scenarios.empty()) return false;
    const uint32_t N = (uint32_t)scenarios.size() * (uint32_t)nF;
    const size_t row = (size_t)L * (size_t)n_embd;
    pos.assign((size_t)N * row, 0.0f);
    neg.assign((size_t)N * row, 0.0f);

    int n_slots = capture_slots < 1 ? 1 : capture_slots;
    int n_ctx   = capture_ctx < 64 ? 64 : capture_ctx;

    std::vector<llama_context *> ctxs((size_t)n_slots, nullptr);
    for (int i = 0; i < n_slots; ++i) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)n_ctx;
        cp.type_k = g_cache_type_k; cp.type_v = g_cache_type_v;   // -ctk/-ctv (default q8_0)
        cp.n_batch = (uint32_t)n_ctx;   // capture is a single short prefill; keep the compute buffer sized to it (not the 2048 default)
        ctxs[(size_t)i] = llama_init_from_model(model, cp);
        if (!ctxs[(size_t)i]) {
            std::fprintf(stderr, "capture: context %d/%d init failed (out of VRAM?) -> aborting\n", i + 1, n_slots);
            for (auto * c : ctxs) if (c) llama_free(c);
            return false;
        }
    }

    std::atomic<uint32_t> next_pair(0);
    std::atomic<bool>     failed(false);
    std::atomic<uint32_t> done_count(0);
    std::mutex print_mu;
    auto worker = [&](int slot) {
        llama_context * c = ctxs[(size_t)slot];
        for (uint32_t idx = next_pair.fetch_add(1); idx < N; idx = next_pair.fetch_add(1)) {
            const uint32_t s = idx / (uint32_t)nF;
            const int f = (int)(idx % nF);
            std::vector<float> pa = capture_pair(model, c, tmpls, pos_facets[f], scenarios[s], L, n_embd);
            if (pa.empty()) { failed = true; return; }
            std::copy(pa.begin(), pa.end(), pos.data() + (size_t)idx * row);
            std::vector<float> na = capture_pair(model, c, tmpls, neg_facets[f], scenarios[s], L, n_embd);
            if (na.empty()) { failed = true; return; }
            std::copy(na.begin(), na.end(), neg.data() + (size_t)idx * row);
            const uint32_t d = done_count.fetch_add(1) + 1;
            if (d % 40 == 0 || d == N) { std::lock_guard<std::mutex> lk(print_mu); std::printf("      captured %u/%u pairs\n", d, N); }
        }
    };

    if (n_slots > 1) {
        std::vector<std::thread> ts; ts.reserve((size_t)n_slots);
        for (int i = 0; i < n_slots; ++i) ts.emplace_back(worker, i);
        for (auto & t : ts) t.join();
    } else {
        worker(0);
    }

    for (auto * c : ctxs) if (c) llama_free(c);
    return !failed.load();
}

// Stage 2+3 core: capture all pairs + build the CV gguf (caller loads model/tmpls).
static bool generate_cv_core(llama_model * model, const common_chat_templates * tmpls,
        const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
        const std::vector<std::string> & scenarios, int n_embd, uint32_t L, const Params & sp) {
    const int nF = (int)pos_facets.size();
    if ((int)neg_facets.size() != nF) std::fprintf(stderr, "warn: %zu pos vs %zu neg facets\n", pos_facets.size(), neg_facets.size());
    const uint32_t N = (uint32_t)scenarios.size() * (uint32_t)nF;
    std::printf("    %zu scenarios x %d facets = %u pairs per side\n", scenarios.size(), nF, N);

    std::vector<float> pos, neg;

    { UcvgsStageScope _cap("Capture activations");
        std::printf("      capture: %d parallel slot(s), per-slot n_ctx=%d\n", sp.capture_slots, sp.capture_ctx);
        if (!capture_all(model, tmpls, pos_facets, neg_facets, scenarios, n_embd, L, sp.capture_ctx, sp.capture_slots, pos, neg)) {
            std::fprintf(stderr, "capture failed\n"); return false;
        }
        std::printf("    captured all %u pairs\n", N);
    }

    { UcvgsStageScope _diag("Compute layer diagnostics");
        const bool ok = run(sp, N, L, (uint32_t)n_embd, pos, neg);   // run() already prints the detailed "Wrote ... (N layers, d=...)"
        return ok;
    }
}

// resumable end-to-end pipeline. Layout <out-dir>/<slug>/{stimuli.json,<slug>.gguf}, skip-if-exists per stage.
// Stages 1 + 2/3 (skip-if-exists)
static bool pipeline_stages_123(llama_model * model_p, llama_context * ctx, const common_chat_templates * tmpls,
                                std::vector<std::string> pos_facets, std::vector<std::string> neg_facets,
                                std::string scenario_desc, int pairs_per_slot, float reserve, int max_attempts,
                                const Params & sp, const std::string & stimuli_path, const std::string & gguf_path, uint32_t seed) {
    { UcvgsStageScope _gp("Generation phase");
        if (file_exists(stimuli_path)) {
            std::printf("Reusing existing stimuli: %s\n", to_slash(stimuli_path).c_str());
        } else {
            UcvgsStageScope _sg("Scenario generation");
            if (scenario_desc.empty()) scenario_desc = "general day-to-day situations";
            std::printf("\nGenerating stimuli\n");
            std::mt19937 rng(seed ? seed : (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());
            const auto stimuli = make_stimuli_core(model_p, ctx, tmpls, pos_facets, neg_facets, scenario_desc, pairs_per_slot, reserve, max_attempts, rng);
            if (stimuli.empty()) { std::fprintf(stderr, "ucvg: no scenarios generated\n"); return false; }
            if (!write_stimuli_json(stimuli_path, stimuli)) return false;
        }
    }

    { UcvgsStageScope _vg("Vector generation");
        if (file_exists(gguf_path)) {
            std::printf("Reusing existing control vector: %s\n", to_slash(gguf_path).c_str());
        } else {
            std::vector<std::string> scenarios = read_trained_scenarios(stimuli_path);
            if (scenarios.empty()) { std::fprintf(stderr, "ucvg: no trained scenarios in %s\n", stimuli_path.c_str()); return false; }
            std::printf("\nBuilding CV from %zu training scenarios\n", scenarios.size());
            const int n_embd = llama_model_n_embd(model_p);
            const uint32_t L = (uint32_t)llama_model_n_layer(model_p);
            if (!generate_cv_core(model_p, tmpls, pos_facets, neg_facets, scenarios, n_embd, L, sp)) return false;
        }
    }
    return true;
}


// ---- Stage 4/5: evaluate a control vector (--evaluate) ----

static std::vector<std::string> read_heldout_scenarios(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::vector<std::string> out;
    std::string line;
    auto unesc = [](const std::string & s) {
        std::string o; for (size_t i = 0; i < s.size(); ++i) { char c = s[i]; if (c == '\\' && i + 1 < s.size()) { char n = s[++i]; o += (n == 'n') ? '\n' : (n == 'r') ? '\r' : (n == 't') ? '\t' : n; } else o += c; }
        return o;
    };
    while (std::getline(f, line)) {
        size_t k = line.find("\"scenario\""); if (k == std::string::npos) continue;
        size_t i = line.find(':', k); if (i == std::string::npos) continue; ++i;
        while (i < line.size() && line[i] != '"') ++i;
        if (i >= line.size()) continue;
        ++i;
        std::string raw;
        while (i < line.size()) { char c = line[i]; if (c == '\\' && i + 1 < line.size()) { raw += c; ++i; raw += line[i]; } else if (c == '"') break; else raw += c; ++i; }
        std::string scenario = unesc(raw);
        bool trained = true; size_t tk = line.find("\"trained_on\"");
        if (tk != std::string::npos) { size_t j = line.find(':', tk); if (j != std::string::npos) { ++j; while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) ++j; trained = (line.compare(j, 4, "true") == 0); } }
        if (!trained && !scenario.empty()) out.push_back(scenario); // held-out only
    }
    return out;
}

// Apply a control vector at the given scale (strength); scale 0 clears it.
static bool apply_cv(llama_context * ctx, const llama_model * model, const std::string & cv_gguf, float scale) {
    if (scale == 0.0f) { llama_set_adapter_cvec(ctx, NULL, 0, 0, 0, 0); return true; }
    common_control_vector_data cv = common_control_vector_load({ { scale, cv_gguf } });
    if (cv.data.empty() || cv.n_embd <= 0) { std::fprintf(stderr, "ucvg --evaluate: CV load failed (%s)\n", cv_gguf.c_str()); return false; }
    if (llama_set_adapter_cvec(ctx, cv.data.data(), cv.data.size(), cv.n_embd, 1, llama_model_n_layer(model)) != 0) { std::fprintf(stderr, "ucvg --evaluate: set_adapter_cvec failed\n"); return false; }
    return true;
}

static bool is_dig(char c) { return c >= '0' && c <= '9'; }
static std::vector<int> parse_int_array(const std::string & s, const std::string & key) {
    std::vector<int> out; size_t k = s.find(key); if (k == std::string::npos) return out;
    size_t b = s.find('[', k); if (b == std::string::npos) return out;
    size_t e = s.find(']', b);
    size_t end = (e == std::string::npos) ? s.size() : e;   // tolerate a missing closing bracket (truncated judge output)
    std::string inner = s.substr(b + 1, end - b - 1); size_t i = 0;
    while (i < inner.size()) {
        while (i < inner.size() && !(is_dig(inner[i]) || inner[i] == '-')) ++i;
        if (i >= inner.size()) break;
        bool neg = false; if (inner[i] == '-') { neg = true; ++i; }
        long v = 0; bool any = false; while (i < inner.size() && is_dig(inner[i])) { v = v * 10 + (inner[i] - '0'); ++i; any = true; }
        if (any) out.push_back(neg ? -(int)v : (int)v);
    }
    return out;
}
static int parse_int_after(const std::string & s, const std::string & key) {
    size_t k = s.find(key); if (k == std::string::npos) return -1;
    size_t i = s.find(':', k) + 1; while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
    bool neg = false; if (s[i] == '-') { neg = true; ++i; }
    long v = 0; bool any = false; while (i < s.size() && is_dig(s[i])) { v = v * 10 + (s[i] - '0'); ++i; any = true; }
    return any ? (neg ? -((int)v) : (int)v) : -1;
}

// ---- Evaluation report ----
struct UcvgsScenario {
    std::string scenario; int alignment = 0; bool valid = true;
    std::map<std::string, int> raw;              // scale key -> raw coherence
    std::map<std::string, double> norm;          // scale key -> normalized (rel to this scenario's own scale-0)
    std::map<std::string, std::string> comps;    // scale key -> reaction text (the "examples")
    std::string reasoning;
};
struct UcvgsReport {
    std::vector<UcvgsScenario> scenarios;
    std::vector<float> scales;
    double mean_alignment = 0, std_alignment = 0;
    std::vector<int> alignment_scores;
    std::map<std::string, double> mean_norm_per_scale;   // scale key -> normalized mean (autoscale queries this)
    std::map<std::string, double> mean_raw_per_scale;    // scale key -> raw mean
    double overall_norm = 0, overall_raw = 0, quality = 0;
    int n_req = 0; double gen_rate_sum = 0, prompt_rate_sum = 0;
};


// Global eval accounting: accumulated across ALL run_eval_loop calls (autoscale iterations + final eval).
struct UcvgsEvalStats { long long n_reactions = 0, n_judges = 0; double gen_tokens = 0.0, prompt_tokens = 0.0; };
static UcvgsEvalStats g_eval_stats;

// Generate reactions (per scale, CV applied) + judge each scenario; aggregate into a report. No file writes; prints progress only when verbose.
static UcvgsReport run_eval_loop(const llama_model * model_p, llama_context * ctx, const common_chat_templates * tmpls,
        const std::string & trait_name, const std::string & cv_gguf,
        const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
        const std::vector<std::string> & heldout, const std::vector<float> & scales, uint32_t seed, bool verbose = false, bool time_sub = false) {
    UcvgsReport rep; rep.scales = scales;
    std::string pos_p, neg_p;
    for (size_t i = 0; i < pos_facets.size(); ++i) { if (i) pos_p += "\n\n---\n\n"; pos_p += pos_facets[i]; }
    for (size_t i = 0; i < neg_facets.size(); ++i) { if (i) neg_p += "\n\n---\n\n"; neg_p += neg_facets[i]; }
    const std::string eval_sys = "You are a person. React naturally to the situation described.";
    std::mt19937 rng(seed ? seed : (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());

    // Generation phase: reactions per scale, collecting per-request rates for the speed average.
    const int _rg = time_sub ? g_timer.begin("Reaction generation") : -1;
    std::vector<std::map<std::string, std::string>> comps(heldout.size());
    size_t rix = 0;
    for (float s : scales) {
        if (verbose) std::printf("    %zu/%zu Generating reactions at scale %.2f ...\n", ++rix, scales.size(), (double)s);
        if (!apply_cv(ctx, model_p, cv_gguf, s)) continue;
        const std::string key = scale_key(s);
        for (size_t si = 0; si < heldout.size(); ++si) {
            int nt = 0; double sec = 0; int pt = 0;
            comps[si][key] = generate_text(model_p, ctx, tmpls, eval_sys, "Situation: " + heldout[si] + "\nYour reaction:", 100, 0.0f, rng, "eval/reaction", true, &nt, &sec, &pt);
            ++g_eval_stats.n_reactions; g_eval_stats.gen_tokens += nt; g_eval_stats.prompt_tokens += pt;   // global accounting (autoscale + final)
            if (sec > 0) { ++rep.n_req; rep.gen_rate_sum += (double)nt / sec; rep.prompt_rate_sum += (double)pt / sec; }
        }
    }
    if (time_sub) g_timer.finish(_rg);
    apply_cv(ctx, model_p, cv_gguf, 0.0f); // clear the CV before judging

    // Judge phase: one call per scenario over all its completions.
    const int _je = time_sub ? g_timer.begin("Judge evaluation") : -1;
    for (size_t si = 0; si < heldout.size(); ++si) {
        UcvgsScenario sc; sc.scenario = heldout[si];
        std::string jp = "SCENARIO:\n\"" + heldout[si] + "\"\n\nTRAIT: " + trait_name + "\n\nPOSITIVE PERSONA (high trait):\n" + pos_p +
                         "\n\nNEGATIVE PERSONA (low trait):\n" + neg_p + "\n\nACTUAL REACTIONS AT DIFFERENT SCALES:\n";
        for (float s : scales) { char b[48]; snprintf(b, sizeof(b), "- Scale %g: ", s); jp += std::string(b) + "\"" + comps[si][scale_key(s)] + "\"\n"; }
        jp += "\nNow, please evaluate both:\n"
              "(a) the COHERENCE of each individual reaction, 0-100 scale, where\n"
              "    - 100: perfectly clean and on-topic\n"
              "    - 85:  perfectly clean phrasing and grammar, but the response is not on topic\n"
              "    - 70:  awkward phrasing\n"
              "    - 50:  typos/missing endings\n"
              "    - 30:  glitches like doubled words\n"
              "    - 15:  severe repetition/breakdown\n"
              "    - 10:  messy words vaguely related\n"
              "    - 5:   word salad\n"
              "    - 0:   absolute noise\n"
              "(b) the overall ALIGNMENT of the WHOLE GRADIENT to the trait. This is a single integer 0-100, where:\n"
              "    - 100: every step in scale produces a clear, noticeable, monotonic shift in the right direction, with distinct contrast between adjacent scales; no flat segments, most importantly - difference between lowest values and the default state.\n"
              "    - 80:  strong alignment, maybe 1-2 minor inconsistencies\n"
              "    - 60:  noticeable gradient but with some weaknesses (e.g., middle is flat, contrast only clear at extremes)\n"
              "    - 40:  only extreme scales show trait; middle is neutral\n"
              "    - 20:  faint hint of trait at extremes only\n"
              "    - 0:   no discernible pattern or wrong direction\n"
              "\nTruncated responses do NOT count as coherence artifact, because that is caused by token limit on the responses.\n"
              "\nReturn a single JSON object: {\"coherence_scores\": [one score per reaction, same order], \"alignment_score\": integer, \"reasoning\": \"one short sentence\"}.\n"
              "The list of coherence scores MUST have exactly the same number of elements as reactions presented above, in the same scale order. Do not include any other text.";
        for (const auto & kv : comps[si]) sc.comps[kv.first] = kv.second;   // every reaction -> report "examples"

        std::string jt;
        bool parsed = false;
        for (int attempt = 1; attempt <= 2 && !parsed; ++attempt) {
            int jnt = 0, jpt = 0;
            jt = generate_text(model_p, ctx, tmpls, "You are an expert evaluator.", jp, 1024, 0.9f, rng, "eval/judge", true, &jnt, nullptr, &jpt);   // temp 0.9 (not greedy) to avoid token-repetition collapse; quiet: no per-token line in eval/autoscale
            ++g_eval_stats.n_judges; g_eval_stats.gen_tokens += jnt; g_eval_stats.prompt_tokens += jpt;   // global accounting (per attempt)

            const auto arr = parse_int_array(jt, "coherence_scores");
            int ai = parse_int_after(jt, "alignment_score");
            if (!arr.empty() && ai >= 0) {
                parsed = true;
                int j = 0;
                for (float s : scales) { if (j < (int)arr.size()) sc.raw[scale_key(s)] = arr[j]; ++j; }
                sc.alignment = ai;
                sc.reasoning = parse_json_string_field(jt, "reasoning");
                rep.scenarios.push_back(sc);
                if (verbose) std::printf("        %zu/%zu Evaluated scenario\n", si + 1, heldout.size());
            } else if (attempt == 2) {
                // both attempts failed to parse: exclude this scenario from the score aggregates (not pushed) and print the full response.
                std::printf("        %zu/%zu FAILED TO PARSE judge response (excluded from scores)\n", si + 1, heldout.size());
                std::printf("        scenario: \"%s\"\n", sc.scenario.c_str());
                std::printf("-------- full judge response --------\n%s\n----------------------------------------\n", jt.c_str());
            }
        }
    }
    if (time_sub) g_timer.finish(_je);

    // Per-scenario normalization relative to each scenario's own scale-0 baseline
    for (auto & sc : rep.scenarios) {
        double base = 0; auto it = sc.raw.find(scale_key(0.0f)); if (it != sc.raw.end()) base = it->second;
        for (const auto & kv : sc.raw) sc.norm[kv.first] = (base > 0) ? std::min(100.0, (double)kv.second / base * 100.0) : (double)kv.second;
    }

    // Aggregation over valid scenarios
    std::map<std::string, double> raw_sum, norm_sum; std::map<std::string, int> raw_cnt, norm_cnt;
    double align_sum = 0; int n_valid = 0;
    for (const auto & sc : rep.scenarios) { if (!sc.valid) continue; ++n_valid; align_sum += sc.alignment; rep.alignment_scores.push_back(sc.alignment);
        for (const auto & kv : sc.raw)  { raw_sum[kv.first]  += kv.second; ++raw_cnt[kv.first]; }
        for (const auto & kv : sc.norm) { norm_sum[kv.first] += kv.second; ++norm_cnt[kv.first]; }
    }
    rep.mean_alignment = n_valid ? align_sum / n_valid : 0.0;
    double a2 = 0; for (int v : rep.alignment_scores) a2 += (double)v * v;
    rep.std_alignment = n_valid ? std::sqrt(std::max(0.0, a2 / n_valid - rep.mean_alignment * rep.mean_alignment)) : 0.0;
    for (float s : scales) { const std::string k = scale_key(s);
        rep.mean_raw_per_scale[k]  = raw_cnt[k]  ? raw_sum[k]  / raw_cnt[k]  : 0.0;
        rep.mean_norm_per_scale[k] = norm_cnt[k] ? norm_sum[k] / norm_cnt[k] : 0.0; }
    double on = 0, oraw = 0; for (float s : scales) { on += rep.mean_norm_per_scale[scale_key(s)]; oraw += rep.mean_raw_per_scale[scale_key(s)]; }
    if (!scales.empty()) { on /= (double)scales.size(); oraw /= (double)scales.size(); }
    rep.overall_norm = on; rep.overall_raw = oraw;
    rep.quality = on * rep.mean_alignment / 100.0;
    return rep;
}

// Stage 4/5 on an already-loaded model+context: per-scale reactions (CV applied), judge each, aggregate a report
static bool evaluate_core(const llama_model * model_p, llama_context * ctx, const common_chat_templates * tmpls,
                         const std::string & trait_name, const std::string & cv_gguf,
                         const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
                         std::vector<std::string> heldout, const std::vector<float> & scales, int eval_max,
                         const std::string & out, uint32_t seed, double * quality_out = nullptr) {
    if (heldout.empty()) { std::fprintf(stderr, "ucvg --evaluate: no held-out scenarios\n"); return false; }
    if ((int)heldout.size() > eval_max) heldout.resize(eval_max);
    std::printf("\nEvaluating (%zu scenarios for each of the %zu scales)\n", heldout.size(), scales.size());
    UcvgsReport rep = run_eval_loop(model_p, ctx, tmpls, trait_name, cv_gguf, pos_facets, neg_facets, heldout, scales, seed, true, true);   // verbose + time sub-phases (final eval)

    if (rep.n_req > 0) {
        std::printf("\n    Average generation speeds over %d API calls:\n", rep.n_req);
        std::printf("        prompt_per_second:   %.1f\n", rep.prompt_rate_sum / rep.n_req);
        std::printf("        predicted_per_second: %.1f\n\n", rep.gen_rate_sum / rep.n_req);
    }

    auto f2 = [](double x) { char b[32]; snprintf(b, sizeof(b), "%.2f", x); return std::string(b); };   // round to 2 dp
    const std::vector<float> ss = [&] { std::vector<float> v = rep.scales; std::sort(v.begin(), v.end()); return v; }();   // numeric ascending

    UcvgsStageScope _save("Save report");
    std::ofstream jf(out);
    if (!jf) { std::fprintf(stderr, "ucvg --evaluate: cannot write %s\n", out.c_str()); return false; }
    auto emit_map = [&](const std::map<std::string, double> & m) {
        bool first = true; jf << "{\n";
        for (float s : ss) { const std::string k = scale_key(s); auto it = m.find(k); if (it == m.end()) continue;
            if (!first) jf << ",\n";
            jf << "      \"" << json_escape(k) << "\": " << f2(it->second);
            first = false;
        }
        jf << "\n    }";
    };

    jf << "{\n";
    jf << "  \"trait\": \"" << json_escape(trait_name) << "\",\n";
    { bool cf = true; jf << "  \"config\": { \"scales\": [ "; for (float s : ss) { if (!cf) jf << ", "; jf << scale_key(s); cf = false; } jf << " ], \"stimuli_count\": " << heldout.size() << " },\n"; }
    jf << "  \"per_scenario\": [\n";
    for (size_t i = 0; i < rep.scenarios.size(); ++i) {
        const auto & sc = rep.scenarios[i];
        jf << "    {\n";
        jf << "      \"scenario\": \"" << json_escape(sc.scenario) << "\",\n";
        jf << "      \"alignment_score\": " << sc.alignment << ",\n";
        jf << "      \"reasoning\": \"" << json_escape(sc.reasoning) << "\",\n";
        jf << "      \"completions\": {\n";
        bool cf = true;
        for (float s : ss) { const std::string k = scale_key(s); auto it = sc.comps.find(k); if (it == sc.comps.end()) continue;
            int score = 0; auto rit = sc.raw.find(k); if (rit != sc.raw.end()) score = rit->second;   // raw per-reaction coherence
            if (!cf) jf << ",\n";
            jf << "        \"" << json_escape(k) << "\": [" << score << ", \"" << json_escape(it->second) << "\"]";
            cf = false;
        }
        jf << "\n      }\n";
        jf << (i + 1 < rep.scenarios.size() ? "    },\n" : "    }\n");
    }
    jf << "  ],\n  \"aggregate\": {\n";
    jf << "    \"mean_alignment\": " << f2(rep.mean_alignment) << ",\n";
    jf << "    \"std_alignment\": " << f2(rep.std_alignment) << ",\n";
    jf << "    \"alignment_scores\": [\n";
    for (size_t i = 0; i < rep.alignment_scores.size(); ++i) jf << "      " << rep.alignment_scores[i] << (i + 1 < rep.alignment_scores.size() ? ",\n" : "\n");
    jf << "    ],\n";
    jf << "    \"mean_coherence_per_scale\": "; emit_map(rep.mean_norm_per_scale); jf << ",\n";
    jf << "    \"overall_mean_coherence\": " << f2(rep.overall_norm) << ",\n";
    jf << "    \"mean_raw_coherence_per_scale\": "; emit_map(rep.mean_raw_per_scale); jf << ",\n";
    jf << "    \"overall_raw_mean_coherence\": " << f2(rep.overall_raw) << ",\n";
    jf << "    \"overall_quality_score\": " << f2(rep.quality) << "\n";
    jf << "  }\n}\n";
    if (quality_out) *quality_out = rep.quality;   // unified tool prints its own final summary
    else { std::printf("\nOverall quality score: %.2f\n", rep.quality); std::printf("report at %s\n", to_slash(out).c_str()); }
    return true;
}

// ---- Autoscale: iterative boundary search + evenly-spaced scale list ----
static const double AS_TARGET = 75.0, AS_WINDOW = 0.10, AS_FIRST_PROBE = 0.3, AS_STEP = 0.4;
static const int    AS_MAX_ITERS = 8;
static const double AS_MIN_SCALE = 0.005, AS_MAX_SCALE = 8.0;
static const int    AS_SUBSET_SIZE = 8;

// Evenly-spaced scale list from -neg to +pos (SUBSTEP_RATE granularity), matching _generate_scale_list.

// Probe both sides adaptively until normalized coherence lands in target*(1+-window); returns the eval scale grid.
static std::vector<float> autoscale_scales(const llama_model * model_p, llama_context * ctx, const common_chat_templates * tmpls,
        const std::string & trait_name, const std::string & cv_gguf,
        const std::vector<std::string> & pos_facets, const std::vector<std::string> & neg_facets,
        const std::vector<std::string> & heldout, uint32_t seed) {
    std::printf("\nAutoscaling (target: %.1f +/- %.0f%%; max iterations: %d)\n", AS_TARGET, AS_WINDOW * 100.0, AS_MAX_ITERS);
    std::vector<std::string> subset = heldout; if ((int)subset.size() > AS_SUBSET_SIZE) subset.resize(AS_SUBSET_SIZE);

    float neg_abs = AS_FIRST_PROBE, pos_abs = AS_FIRST_PROBE;
    bool neg_done = false, pos_done = false;
    std::vector<std::pair<float, double>> neg_hist, pos_hist;
    const double lower = AS_TARGET * (1 - AS_WINDOW), upper = AS_TARGET * (1 + AS_WINDOW);

    double neg_coh = 0.0, pos_coh = 0.0;
    for (int i = 0; i < AS_MAX_ITERS; ++i) {
        if (neg_done && pos_done) break;
        std::vector<float> scales = { 0.0f };
        if (!neg_done) scales.push_back(-neg_abs);
        if (!pos_done) scales.push_back(pos_abs);
        UcvgsReport rep = run_eval_loop(model_p, ctx, tmpls, trait_name, cv_gguf, pos_facets, neg_facets, subset, scales, seed, false);
        auto get = [&](float s) { auto it = rep.mean_norm_per_scale.find(scale_key(s)); return it != rep.mean_norm_per_scale.end() ? it->second : 0.0; };
        if (!neg_done) { neg_coh = get(-neg_abs); neg_hist.push_back({ -neg_abs, neg_coh });
            if (lower <= neg_coh && neg_coh <= upper) neg_done = true;
            else neg_abs = (float)std::max(AS_MIN_SCALE, std::min(AS_MAX_SCALE, neg_abs * (1 + AS_STEP * (neg_coh / AS_TARGET - 1)))); }
        if (!pos_done) { pos_coh = get(pos_abs); pos_hist.push_back({ pos_abs, pos_coh });
            if (lower <= pos_coh && pos_coh <= upper) pos_done = true;
            else pos_abs = (float)std::max(AS_MIN_SCALE, std::min(AS_MAX_SCALE, pos_abs * (1 + AS_STEP * (pos_coh / AS_TARGET - 1)))); }

        auto pct = [](double ratio) { char b[64]; if (ratio > 1) snprintf(b, sizeof(b), "%.2f%% over target", (ratio - 1) * 100); else snprintf(b, sizeof(b), "%.2f%% of target", ratio * 100); return std::string(b); };
        std::printf("    Iteration %d: pos: %s, neg: %s\n", i, pct(pos_coh / AS_TARGET).c_str(), pct(neg_coh / AS_TARGET).c_str());
    }

    const std::vector<float> scales = generate_scale_list(neg_abs, pos_abs);
    (void)neg_hist; (void)pos_hist;   // probe history no longer printed
    std::printf("    generated scales: [ "); for (size_t i = 0; i < scales.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", (double)scales[i]); } std::printf(" ]\n");
    return scales;
}


// Full help for the unified tool (-h / --help). Stages are named (not numbered); keys in "short  long  description".
static void print_ucvg_help() {
    std::fputs(
R"UCVG(A tool for creating the control knobs (vectors) for ai response steering.

Stages (each is resumable: re-running skips any stage whose output file already exists)
  stimuli generation      -> <out-dir>/<trait>/stimuli.json
  control vector build    -> <out-dir>/<trait>/<trait>.gguf
  evaluation              -> <out-dir>/<trait>/eval_report.json   (on by default; -e off to skip)

Standard llama.cpp keys
  -m  --model             path to a .gguf model
   -ngl --gpu-layers / --n-gpu-layers   GPU layers to offload (default: -1 = all; lower to fit smaller VRAM)
   -ctk --cache-type-k     KV-cache K type (default: q8_0): f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1
   -ctv --cache-type-v     KV-cache V type (default: q8_0): same options as -ctk

Required keys (along with -m from the above)
  -t  --trait-name        trait name; names the output subdirectory + display + evaluation anchor
  facets (equal counts), via repeatable -p/-n OR the two files:
    -p  --positive-persona / --pos-facet    positive persona description. For complex personas repeat this key with different descriptions
    -n  --negative-persona / --neg-facet    negative persona/facet text. For complex personas repeat this key with different descriptions
      --pos-facets-file                     one positive facet per line. Use either -p or this key.
      --neg-facets-file                     one negative facet per line
    The amount of positive and negative traits must match.

Allowed keys
  -o   --output-dir / --out-dir            output directory (default: ucvg_out)
  -s   --scenario / --scenario-desc        scenario-direction hint for generation
       --pairs-per-slot                    scenarios per facet slot (default: 60)
       --reserve-for-eval / --max-reserve-for-eval   held-out fraction (default: 0.15); count = round(fraction x total)
       --max-attempts / --stimuli-gen-max-attempts   max generation attempts per slot (default: 3)
       --seed                              RNG seed (0 = time-based)
  -e   --do-eval [on|off]                  enable/disable evaluation (default: on)
       --eval-scales / --scales            explicit comma-separated strictly-increasing scales (overrides autoscale)
       --eval-max / --max-eval-stimuli     max held-out stimuli used for eval (default: 20)
  -asc --auto-scale / --auto-scales        request autoscaling (already the default scale source)
  direction:
       --pca                             use PC1 (power iteration) as the steering direction instead
    -md  --mean-diff                     use the raw mean-diff direction (default)
  method knobs:
       --threshold-frac / --layer-cutoff-frac   layer-selection cutoff fraction (default: 0.0 = all layers are used)
       --n-bootstrap                        bootstrap resamples (default: 100)
       --subsample-frac                     subsample fraction (default: 0.4)
       --consistency-threshold              consistency threshold (default: 1.0)
       --sigma / --metric-smoothing-sigma   metric smoothing sigma (default: 1.0)
       --weight-power / --metric-weight-power   metric weight exponent (default: 1.0)
       --blend / --metric-blend             metric-vs-depth-envelope blend (default: 0.5)
       --depth-envelope                     enable depth-envelope weighting (default: false)
  parallel capture:
       --capture-slots / --slots          parallel capture contexts (default: 1)
       --capture-ctx / --slot-ctx         per-slot context size for capture (default: 512)
   output / debugging:
       --output-model-responses          print the model's full raw response at every generation site (no trimming; special tokens inline)

Resumability + saved files
  Every stage checks whether its output file already exists and reuses it instead of regenerating, so an
  interrupted run can simply be re-run to continue. All outputs live under <out-dir>/<trait>/:
    stimuli.json      the generated scenario list; each entry is {"scenario": ..., "trained_on": true|false}
                      marking training vs held-out-for-evaluation scenarios
    <trait>.gguf       the built control vector: per-layer steering vectors
    eval_report.json  per-scenario coherence/alignment scores + aggregate metrics

How to describe personas
  Describe behavior concretely in the second person ("You are ..."), not a bare label. Just like a system prompt
  For a complex trait, break up the descriptions - repeat -p/-n, or one facet per line in the file.
  Examples (positive vs negative):
    Verbosity:    "You are extremely verbose and expansive; you elaborate on every answer."   vs   "You are terse and concise; as few words as possible."
    Openness:     "You are intellectually curious and enjoy abstract, imaginative ideas."      vs   "You are practical and grounded; you stick to concrete facts."
    Neuroticism:  "You are prone to worry and easily anxious."                                 vs   "You are calm, patient and even-tempered."

  Facets file format (pos.txt) -- one persona/facet per line, blank lines ignored:
    You are extremely verbose and expansive; you elaborate on every answer.
    You add background context and explore multiple angles before answering.
    You prefer long, detailed explanations over short ones.

Examples
  path/to/llama-ucvg.exe -m /models/gemma.gguf -t Openness-to-experience -o path/to/output -p "..." -n "..."
  path/to/llama-ucvg.exe -m /models/qwen.gguf -t Neuroticism -o out -md --pos-facets-file pos.txt --neg-facets-file neg.txt
  path/to/llama-ucvg.exe -m /models/llama.gguf -t X -o out -p "..." -n "..." -e off --seed 42
  path/to/llama-ucvg.exe -m /models/gptoss.gguf -t X -o out -p "..." -n "..." --eval-scales -0.5,0.0,0.5 --capture-slots 3
)UCVG", stdout);
}

// The unified tool: load the model ONCE and run stage 1 -> 2/3 (always), then stage 4/5 when -e / -asc.
static int run_all(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::string model, trait_name, pos_f, neg_f, scenario_desc;
    std::string out_dir = "ucvg_out";
    std::vector<std::string> pos_flags, neg_flags; // repeatable -p / -n
    int pairs_per_slot = 60; int max_attempts = 3; float reserve = 0.15f; uint32_t seed = 0;
    bool eval_enabled = true;      // -e is ON by default; "-e off" disables evaluation
    bool auto_scale = false;       // -asc present (requests autoscaling; also forces eval if -e off)
    std::string manual_scales_csv; // --eval-scales <v1,v2,...> (strictly increasing, validated before anything loads)
    int eval_max = 20;
    float threshold_frac = 0.0f, subsample_frac = 0.4f, consistency_threshold = 1.0f;
    float sigma = 1.0f, weight_power = 1.0f, blend = 0.5f; bool depth_envelope = false; int n_bootstrap = 100;
    bool use_pca = false;   // Mean-diff is the default steering direction; --pca opts in to PC1 (power iteration)
    int capture_slots = 1;  // --capture-slots: parallel contexts for stage-2/3 capture (1 = serial)
    int capture_ctx = 512;  // --capture-ctx: per-slot n_ctx for capture (prefill-only)

    auto val = [&](int & i, const char * n, std::string & o) -> bool { if (i + 1 >= argc) { std::fprintf(stderr, "ucvg: missing value for %s\n", n); return false; } o = argv[++i]; return true; };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help")                        { print_ucvg_help(); return 0; }
        else if (a == "-m" || a == "--model")                  { std::string v; val(i, a.c_str(), v); model = v; }
        else if (a == "-t" || a == "--trait-name")             { std::string v; val(i, a.c_str(), v); trait_name = v; }
        else if (a == "-p" || a == "--positive-persona" || a == "--pos-facet") { std::string v; val(i, a.c_str(), v); pos_flags.push_back(v); }
        else if (a == "-n" || a == "--negative-persona" || a == "--neg-facet") { std::string v; val(i, a.c_str(), v); neg_flags.push_back(v); }
        else if (a == "--pos-facets-file")                     { std::string v; val(i, a.c_str(), v); pos_f = v; }
        else if (a == "--neg-facets-file")                     { std::string v; val(i, a.c_str(), v); neg_f = v; }
        else if (a == "-o" || a == "--output-dir" || a == "--out-dir") { std::string v; val(i, a.c_str(), v); out_dir = v; }
        else if (a == "-s" || a == "--scenario" || a == "--scenario-desc") { std::string v; val(i, a.c_str(), v); scenario_desc = v; }
        else if (a == "--pairs-per-slot")                      { std::string v; val(i, a.c_str(), v); pairs_per_slot = std::stoi(v); }
        else if (a == "--reserve-for-eval" || a == "--max-reserve-for-eval") { std::string v; val(i, a.c_str(), v); reserve = std::stof(v); }
        else if (a == "--max-attempts" || a == "--stimuli-gen-max-attempts") { std::string v; val(i, a.c_str(), v); max_attempts = std::stoi(v); }
        else if (a == "--seed")                                { std::string v; val(i, a.c_str(), v); seed = (uint32_t)std::stoul(v); }
        else if (a == "-e" || a == "--do-eval") {
            bool on = true; // bare -e means "on"
            if (i + 1 < argc) { const std::string v = argv[i + 1];
                if      (v == "on"||v == "true"||v == "yes"||v == "y"||v == "1")  { on = true;  ++i; }
                else if (v == "off"||v == "false"||v == "no"||v == "n"||v == "0") { on = false; ++i; }
            }
            eval_enabled = on; // "-e off" disables; bare -e / "-e <next-flag>" stays on
        }
        else if (a == "-asc" || a == "--auto-scale" || a == "--auto-scales") auto_scale = true;
        else if (a == "--eval-scales" || a == "--scales")                   { std::string v; val(i, a.c_str(), v); manual_scales_csv = v; }
        else if (a == "--eval-max" || a == "--max-eval-stimuli") { std::string v; val(i, a.c_str(), v); eval_max = std::stoi(v); }
        // tinkerer (method) knobs
        else if (a == "--threshold-frac" || a == "--layer-cutoff-frac") { std::string v; val(i, a.c_str(), v); threshold_frac = std::stof(v); }
        else if (a == "--n-bootstrap")                             { std::string v; val(i, a.c_str(), v); n_bootstrap = std::stoi(v); }
        else if (a == "--subsample-frac")                         { std::string v; val(i, a.c_str(), v); subsample_frac = std::stof(v); }
        else if (a == "--consistency-threshold")                  { std::string v; val(i, a.c_str(), v); consistency_threshold = std::stof(v); }
        else if (a == "--sigma" || a == "--metric-smoothing-sigma")   { std::string v; val(i, a.c_str(), v); sigma = std::stof(v); }
        else if (a == "--weight-power" || a == "--metric-weight-power") { std::string v; val(i, a.c_str(), v); weight_power = std::stof(v); }
        else if (a == "--blend" || a == "--metric-blend")                 { std::string v; val(i, a.c_str(), v); blend = std::stof(v); }
        else if (a == "--depth-envelope")                         depth_envelope = true;
        else if (a == "--pca")                                    use_pca = true;    // opt in to PC1 (power iteration)
        else if (a == "-md" || a == "--mean-diff")                use_pca = false;   // raw mean-diff (already the default)
        else if (a == "--capture-slots" || a == "--slots")        { std::string v; val(i, a.c_str(), v); capture_slots = std::max(1, std::stoi(v)); }
        else if (a == "--capture-ctx" || a == "--slot-ctx")       { std::string v; val(i, a.c_str(), v); capture_ctx = std::max(64, std::stoi(v)); }
        else if (a == "--output-model-responses")                 g_output_model_responses = true;
        // device / memory knobs (applied at every model-load + context-creation site)
        else if (a == "-ngl" || a == "--gpu-layers" || a == "--n-gpu-layers") { std::string v; val(i, a.c_str(), v); g_ngl = std::stoi(v); }
        else if (a == "-ctk" || a == "--cache-type-k")            { std::string v; val(i, a.c_str(), v); ggml_type t; if (!kv_cache_type_from_str(v, t)) { std::fprintf(stderr, "ucvg: unsupported -ctk cache type: %s\n", v.c_str()); return 1; } g_cache_type_k = t; }
        else if (a == "-ctv" || a == "--cache-type-v")            { std::string v; val(i, a.c_str(), v); ggml_type t; if (!kv_cache_type_from_str(v, t)) { std::fprintf(stderr, "ucvg: unsupported -ctv cache type: %s\n", v.c_str()); return 1; } g_cache_type_v = t; }
    }

    if (model.empty() || trait_name.empty()) {
        std::fprintf(stderr, "ucvg: need -m/--model and -t/--trait-name\n"
                             "      facets via repeatable -p/-n or --pos-facets-file/--neg-facets-file\n"
                             "      evaluation runs by default (autoscale); skip with '-e off'; explicit scales via --eval-scales <a,b,c>\n"
                             "      see -h / --help for all options\n");
        return 1;
    }

    if (!pos_flags.empty() && !pos_f.empty()) { std::fprintf(stderr, "ucvg: use either -p inline persona descriptions OR --pos-facets-file, not both.\n"); return 1; }
    if (!neg_flags.empty() && !neg_f.empty()) { std::fprintf(stderr, "ucvg: use either -n inline persona descriptions OR --neg-facets-file, not both.\n"); return 1; }
    std::vector<std::string> pos_facets = pos_flags.empty() ? read_lines(pos_f) : pos_flags;
    std::vector<std::string> neg_facets = neg_flags.empty() ? read_lines(neg_f) : neg_flags;
    if (pos_facets.empty() || neg_facets.empty()) { std::fprintf(stderr, "ucvg: provide facets via -p/-n (equal counts) or --pos-facets-file/--neg-facets-file\n"); return 1; }
    if (pos_facets.size() != neg_facets.size())   { std::fprintf(stderr, "ucvg: count(-n) must == count(-p) (%zu vs %zu)\n", neg_facets.size(), pos_facets.size()); return 1; }

    // Validate --eval-scales up front (strictly increasing) before loading anything.
    std::vector<float> manual_scales; const bool manual_given = !manual_scales_csv.empty();
    if (manual_given) {
        try {
            size_t i = 0; while (i < manual_scales_csv.size()) {
                size_t c = manual_scales_csv.find(',', i); std::string tok = (c == std::string::npos) ? manual_scales_csv.substr(i) : manual_scales_csv.substr(i, c - i);
                size_t b0 = tok.find_first_not_of(" \t"), b1 = tok.find_last_not_of(" \t"); if (b0 != std::string::npos) tok = tok.substr(b0, b1 - b0 + 1);
                if (!tok.empty()) manual_scales.push_back(std::stof(tok));
                if (c == std::string::npos) break;
                i = c + 1;
            }
        } catch (...) { std::fprintf(stderr, "ucvg: --eval-scales must be a comma-separated list of numbers\n"); return 1; }
        if (manual_scales.size() < 2) { std::fprintf(stderr, "ucvg: --eval-scales needs at least 2 values (got %zu)\n", manual_scales.size()); return 1; }
        for (size_t k = 1; k < manual_scales.size(); ++k)
            if (manual_scales[k] <= manual_scales[k - 1]) { std::fprintf(stderr, "ucvg: --eval-scales must be strictly increasing (values only grow); %f !> %f at position %zu\n", manual_scales[k], manual_scales[k-1], k); return 1; }
    }

    // Eval is ON by default. Scale source: --eval-scales if
    // given, else autoscale (autoscale is the default scale source, so -asc adds no independent effect here). "-e off" is
    // authoritative -> no evaluation (matches the reference having no "skip eval" concept to override).
    const bool run_eval = eval_enabled;
    (void)auto_scale;   // -asc is accepted for CLI compatibility, but autoscale is already the default scale source -> no independent effect here

    const std::string slug = slugify(trait_name);
    const std::string trait_dir = out_dir + "/" + slug;
    const std::string stimuli_path = trait_dir + "/stimuli.json";
    const std::string gguf_path = trait_dir + "/" + slug + ".gguf";
    const std::string eval_path = trait_dir + "/eval_report.json";
    UCVG_MKDIR(out_dir.c_str());
    UCVG_MKDIR(trait_dir.c_str());

    // Pre-load run plan (printed before model load so it reads like a "what will happen" summary).
    std::printf("%s\n", use_pca ? "Using PCA for vector extraction (set -md to use the recommended default, mean-diff)"
                                : "Using mean-diff for vector extraction (recommended default; set --pca to use PC1)");
    std::printf("%s\n", file_exists(gguf_path)   ? "Matching CV found in output directory - will reuse."
                                                 : "No matching CV detected in output directory - will generate.");
    std::printf("%s\n", file_exists(stimuli_path) ? "Matching stimuli found in output directory - will reuse."
                                                  : "No matching stimuli detected in output directory - will generate.");
    std::printf("%s\n", run_eval ? "Evaluation enabled - will evaluate. (Set -e off to disable)"
                                 : "Evaluation disabled - skipping evaluation.");
    if (run_eval) std::printf("%s\n", manual_given ? "Manual evaluation scales provided - using them as-is."
                                                   : "Autoscaling is enabled (or evaluation is enabled with no --eval-scales passed) - will approximate usable vector magnitude.");
    std::printf("Starting...\n");

    g_timer.start_pipeline();
    const int _ml = g_timer.begin("Model load & context init");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_ngl;   // -ngl (default -1 = all layers to GPU)
    llama_model * model_p = llama_model_load_from_file(model.c_str(), mp);
    if (!model_p) { std::fprintf(stderr, "ucvg: model load failed\n"); return 1; }
    const int gen_maxtok = std::min(3840, std::max(512, 50 * pairs_per_slot));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max(gen_maxtok + 768, run_eval ? 4096 : 0); // headroom for stage 1/2/3 + eval reactions/judge (judge prompt + output can be several thousand tokens)
    cp.type_k = g_cache_type_k; cp.type_v = g_cache_type_v;   // -ctk/-ctv (default q8_0)
    llama_context * ctx = llama_init_from_model(model_p, cp);
    if (!ctx) { std::fprintf(stderr, "ucvg: context init failed\n"); return 1; }
    auto tmpls = common_chat_templates_init(model_p, "");
    g_timer.finish(_ml);

    Params sp;
    sp.out = gguf_path;
    sp.threshold_fraction = threshold_frac;
    sp.n_bootstrap = n_bootstrap;
    sp.subsample_frac = subsample_frac;
    sp.consistency_threshold = consistency_threshold;
    sp.snr_smoothing_sigma = sigma;
    sp.snr_weight_power = weight_power;
    sp.snr_blend = blend;
    sp.depth_envelope_enabled = depth_envelope;
    sp.use_pca = use_pca;
    sp.capture_slots = capture_slots;
    sp.capture_ctx   = capture_ctx;
    sp.seed          = seed;   // forward --seed to the Stage-3 bootstrap RNG (previously unseeded -> non-reproducible mask)

    // In parallel-capture mode, if stage 1 will be skipped (stimuli present), free the idle shared context so the
    // N per-slot capture contexts get full VRAM headroom. Recreate it afterwards for eval. Stage 1 needs ctx only
    // when stimuli are absent, so this is safe (ctx stays alive exactly when stage 1 uses it).
    const bool parallel_capture = capture_slots > 1 && file_exists(stimuli_path);
    if (parallel_capture) {
        std::printf("ucvg: parallel capture (%d slots) -> freeing shared context for headroom\n", capture_slots);
        llama_free(ctx); ctx = nullptr;
    }

    if (!pipeline_stages_123(model_p, ctx, tmpls.get(), pos_facets, neg_facets, scenario_desc, pairs_per_slot, reserve, max_attempts, sp, stimuli_path, gguf_path, seed)) return 1;

    // Recreate the shared context for eval if we freed it for parallel capture (stage 1 was skipped).
    if (ctx == nullptr && run_eval) {
        llama_context_params cpe = llama_context_default_params();
        cpe.n_ctx = std::max(gen_maxtok + 768, 4096); // headroom for eval reactions/judge (judge prompt + output can be several thousand tokens)
        cpe.type_k = g_cache_type_k; cpe.type_v = g_cache_type_v;   // -ctk/-ctv (default q8_0)
        ctx = llama_init_from_model(model_p, cpe);
        if (!ctx) { std::fprintf(stderr, "ucvg: eval context init failed\n"); return 1; }
    }

    std::vector<float> scales;   // hoisted so the final summary can report the usage range
    double quality = 0.0;
    if (run_eval) {
        g_eval_stats = UcvgsEvalStats{};   // reset accounting for this eval run
        UcvgsStageScope _ep("Evaluation phase");
        std::vector<std::string> heldout = read_heldout_scenarios(stimuli_path);
        if (heldout.empty()) { std::fprintf(stderr, "ucvg: no held-out scenarios for eval (raise --pairs-per-slot or lower --reserve-for-eval)\n"); return 1; }
        if (manual_given) { scales = manual_scales; }
        else              { UcvgsStageScope _as("Autoscale evaluation scales"); scales = autoscale_scales(model_p, ctx, tmpls.get(), trait_name, gguf_path, pos_facets, neg_facets, heldout, seed); }
        { UcvgsStageScope _fe("Full evaluation");
          if (!evaluate_core(model_p, ctx, tmpls.get(), trait_name, gguf_path, pos_facets, neg_facets, heldout, scales, eval_max, eval_path, seed, &quality)) return 1;
        }
    }

    g_timer.end_pipeline();
    g_timer.print_summary();

    // Final summary (paths with forward slashes).
    const double total_wall = g_timer.pipeline_end - g_timer.pipeline_start;
    std::printf("\nDone in %s\n", UcvgsUnifiedTimer::fmt(total_wall).c_str());
    std::printf("  stimuli at   %s\n", to_slash(stimuli_path).c_str());
    std::printf("  vector at    %s   <---\n", to_slash(gguf_path).c_str());
    if (run_eval && !scales.empty()) {
        std::printf("  report at    %s\n", to_slash(eval_path).c_str());
        std::printf("  Recommended usage range approximation: [%.2f, %.2f] (See the report and test manually to understand the actual usable range)\n", (double)scales.front(), (double)scales.back());
        std::printf("  Overall quality score: %.2f\n", quality);
    }
    if (ctx) llama_free(ctx);
    llama_model_free(model_p);
    return 0;
}

} // namespace

// Forward only WARN + ERROR to stderr; drop the DEBUG per-decode "CUDA Graph id N reused" spam and INFO noise.
static void quiet_log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    ggml_log_set(quiet_log_cb, nullptr);
    llama_log_set(quiet_log_cb, nullptr);

    // Default: the unified pipeline tool (stages 1->3, +4/5 with -e / -asc).
    { const int r = run_all(argc, argv); std::_Exit(r); }
}
