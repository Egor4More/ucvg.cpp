#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

// One control-vector slot: a raw direction (per layer) + per-layer center scalar c_l (= mu_l . v_l)
// + an additive offset `alpha` and multiplicative scales. Tensors are allocated once and re-set in place.
struct cvec_slot {
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> dir;        // per layer (raw direction); [0] = nullptr
    std::vector<ggml_tensor *> center_t;   // per layer (1-elem F32 = c_l); [0] = nullptr
    std::vector<float> norm2;              // per layer (||v_l||^2)

    float dpos  = 0.0f;                    // scale_positive - 1
    float dneg  = 0.0f;                    // scale_negative - 1
    float alpha = 0.0f;                    // additive offset (scales the raw direction)

    ggml_tensor * dpos_t = nullptr;        // 1-elem F32 = dpos
    ggml_tensor * dneg_t = nullptr;        // 1-elem F32 = dneg

    bool allocated = false;
};

struct llama_adapter_cvec {
    // Legacy additive API (kept): routes to slot 0 with alpha=1 and both scales=1.
    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

    // Set one control-vector slot (index `idx`): raw direction + per-layer center scalar c_l + additive offset
    // `alpha` + multiplicative scales. Allocates the slot's tensors on first use; later calls re-set in place.
    bool set_slot(
            int idx,
            const llama_model & model,
            const float * dir_data,
            size_t len,
            int32_t n_embd,
            const float * center,
            size_t center_len,
            int32_t il_start,
            int32_t il_end,
            float alpha,
            float scale_positive,
            float scale_negative);

    // Number of slots [0, n) that are active for the next graph build.
    void set_n_active(int n);

    // Apply all active slots to `cur` at layer `il`. Per slot: h' += (k-1)*dev*v + alpha*v where
    // dev = (h.v - c_l)/||v||^2 and k = scale_positive (dev>=0) else scale_negative. Every contribution is
    // computed from the original `cur` so slots stay independent. Covers additive (scales=1), gain (alpha=0), both.
    ggml_tensor * apply_hybrid_all(ggml_context * ctx, ggml_tensor * cur, int il) const;

private:
    bool ensure_slot(size_t idx, const llama_model & model);

    std::vector<cvec_slot> slots;
    int32_t n_active = 0;
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
