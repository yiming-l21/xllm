#pragma once

namespace xllm {

enum class PolicyType {
  NONE = 0,
  FBCACHE,
  TAYLORSEER,
  FBCACHE_WITH_TAYLORSEER
};

struct DiTCacheConfig {
  struct DiTCacheOptions {
    int num_inference_steps = 25;
    int warmup_steps = 0;
  } ditcache;

  struct FBCacheOptions : public DiTCacheOptions {
    float residual_diff_threshold = 0.09;
  } fbcache;

  struct TaylorSeerOptions : public DiTCacheOptions {
    int n_derivatives = 3;
    int skip_interval_steps = 3;
  } taylorseer;

  struct FBCacheWithTaylorSeerOptions : public DiTCacheOptions {
    float residual_diff_threshold = 0.09;
    int n_derivatives = 3;
  } fbcachewithtaylor;

  PolicyType selected_policy = PolicyType::NONE;
};

}  // namespace xllm
