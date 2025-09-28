#pragma once

#include <cmath>
#include <vector>

#include "dit_cache_impl.h"

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class TaylorSeer : public DitCacheImpl {
 public:
  TaylorSeer(int n_derivatives = 2,
             int warmup_steps = 1,
             int skip_interval_steps = 1)
      : n_derivatives_(n_derivatives),
        ORDER_(n_derivatives + 1),
        skip_interval_steps_(skip_interval_steps) {
    reset_cache();
  }

  void init(const DiTCacheConfig& cfg) override {
    num_inference_steps = cfg.taylorseer.num_inference_steps;
    warmup_steps = cfg.taylorseer.warmup_steps;
    n_derivatives_ = cfg.taylorseer.n_derivatives;
    skip_interval_steps_ = cfg.taylorseer.skip_interval_steps;
    ORDER_ = n_derivatives_ + 1;
    reset_cache();
  }

  void reset_cache() {
    dY_prev_.assign(ORDER_, torch::Tensor());
    dY_current_.assign(ORDER_, torch::Tensor());
    valid_prev_.assign(ORDER_, false);
    valid_current_.assign(ORDER_, false);
    current_step = 1;
    last_non_approximated_step_ = 1;
    use_cache = false;
  }

  void mark_step_begin() { ++current_step; }

  std::pair<std::vector<torch::Tensor>, std::vector<bool>>
  approximate_derivative(torch::Tensor Y) {
    std::vector<torch::Tensor> dY(ORDER_);
    std::vector<bool> valid(ORDER_, false);

    dY[0] = Y;
    valid[0] = true;

    int window = current_step - last_non_approximated_step_;
    for (int i = 0; i < n_derivatives_; ++i) {
      if (i < static_cast<int>(valid_prev_.size()) && valid_prev_[i] &&
          current_step > 1 && window != 0 && dY_prev_[i].defined()) {
        dY[i + 1] = (dY[i] - dY_prev_[i]) / static_cast<double>(window);
        valid[i + 1] = true;
      } else {
        break;
      }
    }
    return {dY, valid};
  }

  torch::Tensor approximate_value() {
    int elapsed = current_step - last_non_approximated_step_;
    if (elapsed < 0) {
      return torch::Tensor();
    }
    if (!dY_current_[0].defined()) {
      return torch::Tensor();
    }
    torch::Tensor output = torch::zeros_like(dY_current_[0]);

    for (int i = 0; i < ORDER_; ++i) {
      if (!valid_current_[i]) break;
      double coef =
          (1.0 / factorial(i)) * std::pow(static_cast<double>(elapsed), i);
      output = output + dY_current_[i] * coef;
    }
    return output;
  }

  void update(const torch::Tensor& Y) {
    torch::Tensor Ystore = Y.detach().clone();
    dY_prev_ = dY_current_;
    valid_prev_ = valid_current_;

    if (!dY_current_[0].defined()) {
      for (int i = 0; i < ORDER_; ++i) {
        dY_current_[i] = torch::zeros_like(Ystore);
        valid_current_[i] = false;
      }
      dY_current_[0] = Ystore;
      valid_current_[0] = true;
    } else {
      auto p = approximate_derivative(Ystore);
      dY_current_ = std::move(p.first);
      valid_current_ = std::move(p.second);
    }
    last_non_approximated_step_ = current_step;
  }

  bool on_before_block(const CacheBlockIn& blockin) { return false; }

  CacheBlockOut on_after_block(const CacheBlockIn& blockin) {
    TensorMap out;
    if (contains_key(blockin.tensors, "hidden_states")) {
      out["hidden_states"] =
          get_tensor_or_empty(blockin.tensors, "hidden_states");
    }
    if (contains_key(blockin.tensors, "encoder_hidden_states")) {
      out["encoder_hidden_states"] =
          get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
    }
    return CacheBlockOut(out);
  }

  bool on_before_step(const CacheStepIn& stepin) override {
    current_step = stepin.step_id;
    if (current_step == 1) {
      reset_cache();
      return false;
    }
    if (current_step <= warmup_steps) {
      use_cache = false;
      return false;
    }
    if (((current_step - warmup_steps + 1) % skip_interval_steps_ == 0)) {
      use_cache = false;
    } else {
      use_cache = true;
    }
    return use_cache;
  }

  CacheStepOut on_after_step(const CacheStepIn& stepin) override {
    if (!use_cache) {
      update(stepin.tensors.at("hidden_states"));
      return CacheStepOut(stepin.tensors);
    }
    TensorMap result = {{"hidden_states", approximate_value()}};
    return CacheStepOut(result);
  }

 private:
  bool use_cache = false;
  int n_derivatives_;
  int ORDER_;
  int skip_interval_steps_;
  int last_non_approximated_step_;

  std::vector<torch::Tensor> dY_prev_;
  std::vector<torch::Tensor> dY_current_;
  std::vector<bool> valid_prev_;
  std::vector<bool> valid_current_;

  static double factorial(int k) {
    return std::tgamma(static_cast<double>(k) + 1.0);
  }
};

}  // namespace xllm
