#pragma once
#include <glog/logging.h>

#include "dit_cache_impl.h"

namespace xllm {

class DiTCache {
 public:
  DiTCache(const DiTCache&) = delete;
  DiTCache& operator=(const DiTCache&) = delete;
  DiTCache(DiTCache&&) = delete;
  DiTCache& operator=(DiTCache&&) = delete;

  ~DiTCache() = default;

  static DiTCache& getinstance() {
    static DiTCache ditcache;
    return ditcache;
  }

  bool init(const DiTCacheConfig& cfg) {
    active_cache = create_dit_cache(cfg);
    if (!active_cache) {
      return false;
    }
    active_cache->init(cfg);
    return true;
  }

  bool on_before_block(CacheBlockIn blockin) {
    return active_cache->on_before_block(blockin);
  }

  CacheBlockOut on_after_block(CacheBlockIn blockin) {
    return active_cache->on_after_block(blockin);
  }

  bool on_before_step(CacheStepIn stepin) {
    return active_cache->on_before_step(stepin);
  }

  CacheStepOut on_after_step(CacheStepIn stepin) {
    return active_cache->on_after_step(stepin);
  }

 protected:
  std::unique_ptr<DitCacheImpl> active_cache;
  DiTCache() = default;
};

}  // namespace xllm
