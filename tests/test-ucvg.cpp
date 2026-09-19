// Unit tests for the pure UCVG helpers (tools/ucvg/ucvg-util.cpp). No model or context required,
// so this builds and runs on every platform (including Windows shared-library builds).
#include "ucvg-util.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#undef NDEBUG

static bool near_eq(float a, float b, float tol) { return std::abs(a - b) <= tol; }

int main() {
    // kv_cache_type_from_str: exact match, case-insensitive, rejects unknown/empty types.
    {
        ggml_type t = GGML_TYPE_F32;
        assert(kv_cache_type_from_str("f32",  t) && t == GGML_TYPE_F32);
        assert(kv_cache_type_from_str("Q8_0", t) && t == GGML_TYPE_Q8_0);   // case-insensitive
        assert(kv_cache_type_from_str("bf16", t) && t == GGML_TYPE_BF16);
        assert(!kv_cache_type_from_str("q8_1",  t));   // not in the allowed list
        assert(!kv_cache_type_from_str("bogus", t));
        assert(!kv_cache_type_from_str("",      t));
    }
    // sigmoid_clipped: midpoint, monotonicity, and symmetric tail clipping at +/-50.
    {
        assert(near_eq(sigmoid_clipped(0.0f), 0.5f, 1e-6f));
        assert(sigmoid_clipped(1.0f) > sigmoid_clipped(0.0f) && sigmoid_clipped(0.0f) > sigmoid_clipped(-1.0f));
        assert(near_eq(sigmoid_clipped(1000.0f),  sigmoid_clipped(50.0f),  1e-6f));   // clamped to +50
        assert(near_eq(sigmoid_clipped(-1000.0f), sigmoid_clipped(-50.0f), 1e-6f));   // clamped to -50
        assert(sigmoid_clipped(1000.0f) < 1.0f && sigmoid_clipped(-1000.0f) > 0.0f);
    }
    // sgnf: sign of a value.
    {
        assert(sgnf(0.5f) == 1.0f && sgnf(-0.5f) == -1.0f && sgnf(0.0f) == 0.0f);
    }
    // smooth_curve: size preserved, interior of a flat signal unchanged, a spike spreads without overshooting.
    {
        const std::vector<float> flat(50, 3.0f);
        auto sm = smooth_curve(flat, 1.0f);
        assert(sm.size() == flat.size());
        assert(near_eq(sm[25], 3.0f, 1e-4f));                          // interior, far from the zero-pad boundary
        const std::vector<float> spike = {0, 0, 0, 1, 0, 0, 0};
        auto sp = smooth_curve(spike, 1.0f);
        assert(sp.size() == spike.size());
        assert(sp[3] < 1.0f);                                          // the peak spreads into its neighbours
        for (float v : sp) { assert(std::isfinite(v)); assert(v >= -1e-5f && v <= 1.0f + 1e-5f); }   // convex combo: bounded
    }
    // compute_depth_envelope: normalised peak at the centre, suppressed edges, flat for L<=1.
    {
        auto env = compute_depth_envelope(100, 0.5f, 0.62f, 0.05f);
        assert(env.size() == 100);
        const float mx = *std::max_element(env.begin(), env.end());
        assert(near_eq(mx, 1.0f, 1e-4f));                              // normalised: peak == 1
        const int imax = (int)(std::max_element(env.begin(), env.end()) - env.begin());
        assert(std::abs((float)imax - 0.5f * (100 - 1)) < 3.0f);       // peak near the centre
        assert(env.front() < 0.2f && env.back() < 0.2f);               // edges suppressed
        auto one = compute_depth_envelope(1, 0.5f, 0.62f, 0.05f);
        assert(one.size() == 1 && near_eq(one[0], 1.0f, 1e-6f));       // L<=1 -> flat 1.0
    }
    // compute_pc1: the dominant axis is recovered; ~zero-variance data is rejected.
    {
        const float X[] = {2, 0, 1, 0, -1, 0, -2, 0};                 // 4 rows x 2 dims, already centred
        std::vector<float> pc;
        assert(compute_pc1(X, 4, 2, pc, 50, 1e-6));
        assert(pc.size() == 2);
        assert(std::abs(pc[0]) > 0.99f && std::abs(pc[1]) < 0.02f);    // PC1 ~= [+-1, 0]
        const float Z[] = {0, 0, 0, 0, 0, 0};
        std::vector<float> zpc;
        assert(!compute_pc1(Z, 3, 2, zpc, 50, 1e-6));                 // ~zero variance -> undefined PC1
    }
    // parse_numbered: dot + paren styles, prose lines ignored, empty input -> empty output.
    {
        auto a = parse_numbered("1. first\n2. second");
        assert(a.size() == 2 && a[0] == "first" && a[1] == "second");
        auto b = parse_numbered("1) one\n2) two");
        assert(b.size() == 2 && b[0] == "one" && b[1] == "two");
        auto c = parse_numbered("intro text\n1. x\ncloser\n2. y\n");
        assert(c.size() == 2 && c[0] == "x" && c[1] == "y");
        assert(parse_numbered("").empty());
    }
    // generate_scale_list: symmetric grid is exact, always contains 0, strictly increasing + unique; degenerate -> {0}.
    {
        auto sym = generate_scale_list(0.5f, 0.5f);
        assert(sym.size() == 11);                                  // -0.5 .. +0.5 in 0.1 steps plus 0
        assert(std::abs(sym.front() - (-0.5f)) < 0.02f);
        assert(std::abs(sym.back()  -  0.5f)  < 0.02f);
        assert(std::find(sym.begin(), sym.end(), 0.0f) != sym.end());
        for (size_t i = 1; i < sym.size(); ++i) assert(sym[i - 1] < sym[i]);   // sorted + unique
        auto z = generate_scale_list(0.0f, 0.0f);
        assert(z.size() == 1 && std::abs(z[0]) < 1e-6f);          // degenerate (no usable magnitude) -> {0}
        auto a = generate_scale_list(0.3f, 0.9f);                 // asymmetric: still a valid grid
        assert(std::find(a.begin(), a.end(), 0.0f) != a.end());
        for (size_t i = 1; i < a.size(); ++i) assert(a[i - 1] < a[i]);
    }
    // scale_key: deterministic %g formatting (no trailing zeros, sign preserved).
    {
        assert(scale_key(0.5f)  == "0.5");
        assert(scale_key(0.0f)  == "0");
        assert(scale_key(-0.25f) == "-0.25");
        assert(scale_key(1.0f)  == "1");
    }
    // parse_json_string_field: exact value, escape decoding, and absent field -> "".
    {
        assert(parse_json_string_field("{\"reasoning\": \"hello world\"}", "reasoning") == "hello world");
        assert(parse_json_string_field("\"reasoning\": \"line1\\nline2\"", "reasoning") == std::string("line1\nline2"));
        assert(parse_json_string_field("{\"other\": 5}", "missing").empty());          // absent field -> ""
    }
    std::printf("test-ucvg: all pure-helper tests passed\n");
    return 0;
}
