#include "ucvg-util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>

namespace {
constexpr float EPS = 1e-12f;      // shared by smooth_curve() and friends (also defined in ucvg.cpp for its own math)
constexpr double AS_SUBSTEP = 0.10; // step size for generate_scale_list() (also defined in ucvg.cpp for autoscale_scales)
}

bool kv_cache_type_from_str(const std::string & s, ggml_type & out) {
    static const std::pair<const char *, ggml_type> allowed[] = {
        {"f32", GGML_TYPE_F32}, {"f16", GGML_TYPE_F16}, {"bf16", GGML_TYPE_BF16},
        {"q8_0", GGML_TYPE_Q8_0}, {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1},
        {"iq4_nl", GGML_TYPE_IQ4_NL}, {"q5_0", GGML_TYPE_Q5_0}, {"q5_1", GGML_TYPE_Q5_1},
    };
    std::string v = s; for (auto & c : v) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    for (const auto & p : allowed) { if (v == p.first) { out = p.second; return true; } }
    return false;
}

float sigmoid_clipped(float x) {
    x = std::max(-50.0f, std::min(50.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float sgnf(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }

// Faithful port of smooth_curve(): zero-pad (1,1), Gaussian kernel over [-r,r] (r = ceil(3*sigma)),
// boundary-clamped window, per-window kernel renormalization.
std::vector<float> smooth_curve(const std::vector<float> & raw_snr, float sigma) {
    const int L = (int)raw_snr.size();
    std::vector<float> padded((size_t)L + 2, 0.0f);
    for (int l = 0; l < L; ++l) padded[(size_t)l + 1] = raw_snr[l];

    sigma = std::max(sigma, EPS);
    const int radius = (int)std::ceil(3.0 * sigma);
    const int n_taps = 2 * radius + 1;
    std::vector<float> kernel(n_taps);
    float ksum = 0.0f;
    for (int i = 0; i < n_taps; ++i) {
        const float x = (float)(i - radius);
        kernel[i] = std::exp(-0.5f * (x / sigma) * (x / sigma));
        ksum += kernel[i];
    }
    for (auto & k : kernel) k /= ksum;

    std::vector<float> smoothed((size_t)L, 0.0f);
    for (int l = 0; l < L; ++l) {
        const int padded_idx = l + 1;
        const int window_start = std::max(0, padded_idx - radius);
        const int window_end = std::min((int)padded.size(), padded_idx + radius + 1);
        const int k_start = radius - (padded_idx - window_start);
        const int wlen = window_end - window_start;
        float num = 0.0f, den = 0.0f;
        for (int j = 0; j < wlen; ++j) {
            num += padded[window_start + j] * kernel[k_start + j];
            den += kernel[k_start + j];
        }
        smoothed[l] = num / den;
    }
    return smoothed;
}

// Faithful port of compute_depth_envelope().
std::vector<float> compute_depth_envelope(int L, float center, float width, float sharpness) {
    std::vector<float> env((size_t)L, 1.0f);
    if (L <= 1) return env;
    if (sharpness <= 0.0f) sharpness = 1e-5f;
    const float left = center - width / 2.0f;
    const float right = center + width / 2.0f;
    float max_env = 0.0f;
    for (int i = 0; i < L; ++i) {
        const float t = (float)i / (float)(L - 1);
        env[i] = sigmoid_clipped((t - left) / sharpness) * sigmoid_clipped((right - t) / sharpness);
        max_env = std::max(max_env, env[i]);
    }
    if (max_env > 1e-12f) { for (auto & e : env) e /= max_env; } else { std::fill(env.begin(), env.end(), 1.0f); }
    return env;
}

bool compute_pc1(const float * Xc, int m, int nd, std::vector<float> & pc1_out, int max_iter, double tol) {
    std::vector<float> v((size_t)nd), y((size_t)m), w((size_t)nd);
    uint64_t s = 0x9E3779B97F4A7C15ull;   // splitmix-style start
    for (int e = 0; e < nd; ++e) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[(size_t)e] = (float)(((double)(s & 0x00FFFFFFull) / 1048576.0) - 0.5); }
    float nv = 0.0f; for (int e = 0; e < nd; ++e) nv += v[(size_t)e] * v[(size_t)e]; nv = std::sqrt(nv);
    if (nv < 1e-9f) { v[0] = 1.0f; nv = 1.0f; }
    for (int e = 0; e < nd; ++e) v[(size_t)e] /= nv;
    for (int it = 0; it < max_iter; ++it) {
        for (int n = 0; n < m; ++n) { const float * xr = &Xc[(size_t)n * nd]; float acc = 0.0f; for (int e = 0; e < nd; ++e) acc += xr[e] * v[(size_t)e]; y[(size_t)n] = acc; }   // y = Xc*v
        std::fill(w.begin(), w.end(), 0.0f);
        for (int n = 0; n < m; ++n) { const float yn = y[(size_t)n]; if (yn == 0.0f) continue; const float * xr = &Xc[(size_t)n * nd]; for (int e = 0; e < nd; ++e) w[(size_t)e] += yn * xr[e]; }   // w = Xc^T*y
        float nw = 0.0f; for (int e = 0; e < nd; ++e) nw += w[(size_t)e] * w[(size_t)e]; nw = std::sqrt(nw);
        if (nw < 1e-9f) return false;   // ~zero variance -> PC1 undefined
        float cosd = 0.0f; for (int e = 0; e < nd; ++e) cosd += v[(size_t)e] * (w[(size_t)e] / nw);   // alignment with the previous direction (->1 when converged)
        for (int e = 0; e < nd; ++e) v[(size_t)e] = w[(size_t)e] / nw;
        if (std::abs(1.0f - cosd) < (float)tol) break;
    }
    pc1_out = std::move(v);
    return true;
}

std::vector<std::string> parse_numbered(const std::string & raw_in) {
    std::string r = raw_in;
    size_t s = 0; while (s < r.size() && (r[s] == '\n' || r[s] == '\r' || r[s] == ' ')) ++s;
    if (r.compare(s, 3, "```") == 0) { size_t nl = r.find('\n', s); if (nl != std::string::npos) r.erase(0, nl + 1); }
    std::vector<std::string> out;
    size_t pos = 0;
    auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    while (pos < r.size()) {
        size_t eol = r.find('\n', pos);
        std::string line = (eol == std::string::npos) ? r.substr(pos) : r.substr(pos, eol - pos);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        size_t ls = 0; while (ls < line.size() && (line[ls] == ' ' || line[ls] == '\t')) ++ls;
        if (ls < line.size()) line = line.substr(ls);
        if (!line.empty() && is_digit(line[0])) {
            size_t k = 0; while (k < line.size() && is_digit(line[k])) ++k;
            if (k > 0 && k < line.size() && (line[k] == '.' || line[k] == ')')) {
                std::string rest = line.substr(k + 1);
                size_t rs = 0; while (rs < rest.size() && (rest[rs] == ' ' || rest[rs] == '\t')) ++rs;
                if (rs < rest.size()) rest = rest.substr(rs);
                if (!rest.empty()) out.push_back(rest);
            }
        }
        if (eol == std::string::npos) break;
        pos = eol + 1;
    }
    return out;
}

// Eval scale list: split each side into AS_SUBSTEP-sized steps, round to 2dp, include 0, de-duplicate and sort.
std::vector<float> generate_scale_list(float neg_abs, float pos_abs) {
    const double full = (double)neg_abs + (double)pos_abs; if (full <= 0) return { 0.0f };
    const double substep = full * AS_SUBSTEP;
    int nc = std::max(1, (int)(neg_abs / substep + 0.5)), pc = std::max(1, (int)(pos_abs / substep + 0.5));
    double ns = neg_abs / nc, ps = pos_abs / pc;
    auto r2 = [](double x) { return (float)(std::round(x * 100.0) / 100.0); };
    std::vector<float> s; for (int i = nc; i >= 1; --i) s.push_back(r2(-ns * i));
    s.push_back(0.0f); for (int i = 1; i <= pc; ++i) s.push_back(r2(ps * i));
    std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
}

std::string scale_key(float s) { char b[32]; snprintf(b, sizeof(b), "%g", s); return b; }

// Tolerant extraction of a JSON string field ("field": "..."). Returns "" if absent.
std::string parse_json_string_field(const std::string & text, const char * field) {
    std::string pat = std::string("\"") + field; size_t p = text.find(pat); if (p == std::string::npos) return "";
    p = text.find(':', p); if (p == std::string::npos) return ""; ++p;
    while (p < text.size() && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r')) ++p;
    if (p >= text.size() || text[p] != '"') return "";
    ++p; std::string out;
    while (p < text.size()) { char c = text[p];
        if (c == '\\' && p + 1 < text.size()) { char n = text[p + 1]; if (n == 'n') out += '\n'; else if (n == 't') out += '\t'; else out += n; p += 2; continue; }
        if (c == '"') break;
        out += c; ++p;
    }
    return out;
}
