#include "dit_cache_impl.h"

#include "fbcache.h"
#include "fbcachewithtaylor.h"
#include "taylorseer.h"

namespace xllm {
std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg) {
  switch (cfg.selected_policy) {
    case PolicyType::FBCACHE:
      return std::make_unique<FBCache>();
    case PolicyType::TAYLORSEER:
      return std::make_unique<TaylorSeer>();
    case PolicyType::FBCACHE_WITH_TAYLORSEER:
      return std::make_unique<FBCacheWithTaylorSeer>();
    default:
      return nullptr;
  }
}
};  // namespace xllm
