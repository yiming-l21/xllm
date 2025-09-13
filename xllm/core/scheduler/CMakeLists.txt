include(cc_library)
include(cc_test)


add_subdirectory(profile)

cc_library(
  NAME 
    scheduler
  HDRS
    chunked_prefill_scheduler.h
    zero_eviction_scheduler.h
    continuous_scheduler.h
    disagg_pd_scheduler.h
    async_response_processor.h
    scheduler.h
    scheduler_factory.h
    decode_priority_queue.h
  SRCS
    chunked_prefill_scheduler.cpp
    zero_eviction_scheduler.cpp
    continuous_scheduler.cpp
    disagg_pd_scheduler.cpp
    async_response_processor.cpp
    scheduler_factory.cpp
  DEPS
    :batch
    :request
    :runtime
    :profile
    glog::glog
    Folly::folly
    absl::time
    absl::synchronization
)


