#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "ggml.h" // ggml_type

// Pure-logic helpers for the UCVG control-vector pipeline, extracted from ucvg.cpp so they can be
// unit-tested (tests/test-ucvg.cpp) without a model or context. No function here touches a model/context.
bool kv_cache_type_from_str(const std::string & s, ggml_type & out);
float sigmoid_clipped(float x);
float sgnf(float x);
std::vector<float> smooth_curve(const std::vector<float> & raw_snr, float sigma);
std::vector<float> compute_depth_envelope(int L, float center, float width, float sharpness);
bool compute_pc1(const float * Xc, int m, int nd, std::vector<float> & pc1_out, int max_iter, double tol);
std::vector<std::string> parse_numbered(const std::string & raw_in);
std::vector<float> generate_scale_list(float neg_abs, float pos_abs);
std::string scale_key(float s);
std::string parse_json_string_field(const std::string & text, const char * field);
