#pragma once
#include <glog/logging.h>

#include "dit_cache_impl.h"
#include "taylorseer.h"

namespace xllm {

class FBCache : public DitCacheImpl {
 public:
  void init(const DiTCacheConfig& cfg) override {
    residual_diff_threshold = cfg.fbcache.residual_diff_threshold;
    num_inference_steps = cfg.fbcache.num_inference_steps;
    warmup_steps = cfg.fbcache.warmup_steps;
    current_step = 0;
  }

  bool on_before_block(const CacheBlockIn& /*blockin*/) override {
    return use_cache;
  }

  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override {
    bool encoder = contains_key(blockin.tensors, "encoder_hidden_states");
    if (blockin.block_id == 0 && encoder) {
      auto hidden_states =
          get_tensor_or_empty(blockin.tensors, "hidden_states");
      auto original_hidden_states =
          get_tensor_or_empty(blockin.tensors, "original_hidden_states");
      auto encoder_hidden_states =
          get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
      auto original_encoder_hidden_states = get_tensor_or_empty(
          blockin.tensors, "original_encoder_hidden_states");

      torch::Tensor first_hidden_states_residual;
      if (hidden_states.defined() && original_hidden_states.defined()) {
        first_hidden_states_residual = hidden_states - original_hidden_states;
      } else {
        first_hidden_states_residual = torch::Tensor();
      }

      bool can_use_cache = false;
      if (first_hidden_states_residual.defined()) {
        can_use_cache = get_use_cache(first_hidden_states_residual);
      }

      torch::Tensor out_hidden = hidden_states;
      torch::Tensor out_encoder_hidden = encoder_hidden_states;

      if (can_use_cache) {
        use_cache = true;
        auto res = apply_prev_hidden_states_residual(hidden_states,
                                                     encoder_hidden_states);
        out_hidden = res.first;
        out_encoder_hidden = res.second;
      } else {
        use_cache = false;
        if (first_hidden_states_residual.defined()) {
          buffers["first_hidden_states_residual"] =
              first_hidden_states_residual;
        }
      }

      TensorMap out_map;
      if (out_hidden.defined()) out_map["hidden_states"] = out_hidden;
      if (out_encoder_hidden.defined())
        out_map["encoder_hidden_states"] = out_encoder_hidden;
      return CacheBlockOut(out_map);
    } else {
      TensorMap out_map;
      if (contains_key(blockin.tensors, "hidden_states"))
        out_map["hidden_states"] = blockin.tensors.at("hidden_states");
      if (encoder)
        out_map["encoder_hidden_states"] =
            blockin.tensors.at("encoder_hidden_states");
      return CacheBlockOut(out_map);
    }
  }

  bool on_before_step(const CacheStepIn& stepin) override {
    current_step = stepin.step_id;
    use_cache = false;
    if (current_step == 1) buffers.clear();
    return false;
  }

  CacheStepOut on_after_step(const CacheStepIn& stepin) override {
    if (!use_cache) {
      torch::Tensor hidden_states =
          get_tensor_or_empty(stepin.tensors, "hidden_states");
      torch::Tensor original_hidden_states =
          get_tensor_or_empty(stepin.tensors, "original_hidden_states");

      if (hidden_states.defined() && original_hidden_states.defined()) {
        auto hidden_states_residual = hidden_states - original_hidden_states;
        buffers["hidden_states_residual"] = hidden_states_residual;
      }

      bool encoder =
          contains_key(stepin.tensors, "encoder_hidden_states") &&
          contains_key(stepin.tensors, "original_encoder_hidden_states");

      if (encoder) {
        auto encoder_hidden_states =
            get_tensor_or_empty(stepin.tensors, "encoder_hidden_states");
        auto original_encoder_hidden_states = get_tensor_or_empty(
            stepin.tensors, "original_encoder_hidden_states");
        if (encoder_hidden_states.defined() &&
            original_encoder_hidden_states.defined()) {
          auto encoder_hidden_states_residual =
              encoder_hidden_states - original_encoder_hidden_states;
          buffers["encoder_hidden_states_residual"] =
              encoder_hidden_states_residual;
        }
      }
    }

    TensorMap out_map;
    if (contains_key(stepin.tensors, "hidden_states"))
      out_map["hidden_states"] = stepin.tensors.at("hidden_states");
    return CacheStepOut(out_map);
  }

 private:
  int skip_interval_steps_ = 3;
  float residual_diff_threshold = 0.09;
  bool use_cache = false;

  std::pair<torch::Tensor, torch::Tensor> apply_prev_hidden_states_residual(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states) {
    auto hidden_states_residual =
        get_tensor_or_empty(buffers, "hidden_states_residual");
    if (hidden_states_residual.defined()) {
      auto new_hidden = hidden_states + hidden_states_residual;
      new_hidden = new_hidden.contiguous();
      return {new_hidden, encoder_hidden_states};
    } else {
      return {hidden_states, encoder_hidden_states};
    }
  }

  bool get_use_cache(const torch::Tensor& first_hidden_states_residual) {
    if (is_in_warmup()) {
      return false;
    }
    if (residual_diff_threshold <= 0.0) {
      return false;
    }
    auto prev_first_hidden_states_residual =
        get_tensor_or_empty(buffers, "first_hidden_states_residual");
    if (!prev_first_hidden_states_residual.defined()) {
      return false;
    }
    bool can_use_cache = is_similar(prev_first_hidden_states_residual,
                                    first_hidden_states_residual,
                                    residual_diff_threshold);
    return can_use_cache;
  }
};

}  // namespace xllm
