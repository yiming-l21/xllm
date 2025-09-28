#pragma once
#include <torch/torch.h>

#include <cmath>

#include "dit_cache_config.h"
#include "dit_cache_data.h"

namespace xllm {

class DitCacheImpl {
 public:
  virtual ~DitCacheImpl() = default;

  virtual void init(const DiTCacheConfig& cfg) = 0;

  virtual bool on_before_block(const CacheBlockIn& blockin) = 0;
  virtual CacheBlockOut on_after_block(const CacheBlockIn& blockin) = 0;

  virtual bool on_before_step(const CacheStepIn& stepin) = 0;
  virtual CacheStepOut on_after_step(const CacheStepIn& stepin) = 0;

 protected:
  int num_inference_steps = 25;
  int warmup_steps = 0;
  int current_step = 0;
  TensorMap buffers;

  bool is_in_warmup() { return current_step <= warmup_steps; }

  static bool contains_key(const TensorMap& m, const std::string& k) {
    return m.find(k) != m.end();
  }

  static torch::Tensor get_tensor_or_empty(const TensorMap& m,
                                           const std::string& k) {
    auto it = m.find(k);
    if (it != m.end()) return it->second;
    return torch::Tensor();
  }

  bool is_similar(const torch::Tensor& t1,
                  const torch::Tensor& t2,
                  float threshold) {
    if (!t1.defined() || !t2.defined()) return false;
    if (threshold <= 0.0f) return torch::allclose(t1, t2);

    if (t1.sizes() != t2.sizes()) return false;

    auto diff = (t1 - t2).abs();
    auto mean_diff_tensor = diff.mean();
    auto mean_t1_tensor = t1.abs().mean();

    double mean_diff_val = mean_diff_tensor.cpu().item<double>();
    double mean_t1_val = mean_t1_tensor.cpu().item<double>();

    if (mean_t1_val == 0.0) {
      const double eps = 1e-6;
      return mean_diff_val < eps;
    }

    double rel = mean_diff_val / mean_t1_val;
    return rel < static_cast<double>(threshold);
  }
};

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg);

}  // namespace xllm
