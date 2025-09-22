/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>

#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>

#include "common/macros.h"
#include "common/types.h"
#include "framework/batch/dit_batch.h"
#include "framework/request/dit_request.h"
#include "scheduler.h"
#include "util/threadpool.h"

namespace xllm {
class DiTEngine;

class DiTAsyncResponseProcessor final {
 public:
  DiTAsyncResponseProcessor() = default;
  ~DiTAsyncResponseProcessor() = default;

  void process_completed_request(std::shared_ptr<DiTRequest> request);
  void process_failed_request(std::shared_ptr<DiTRequest> request,
                              Status status);

 private:
  DISALLOW_COPY_AND_ASSIGN(DiTAsyncResponseProcessor);

  // the threadpool to handle responses
  ThreadPool response_threadpool_;
};

class DiTScheduler : public SchedulerBase {
 public:
  struct Options {
    // the request per batch
    PROPERTY(int32_t, max_request_per_batch) = 4;
  };

  virtual ~DiTScheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<DiTRequest>& request) = 0;
};

class DiTDynamicBatchScheduler : public DiTScheduler {
 public:
  DiTDynamicBatchScheduler(DiTEngine* engine, const Options& options);
  virtual ~DiTDynamicBatchScheduler();

  bool add_request(std::shared_ptr<DiTRequest>& request) override;

  void step(const absl::Duration& timeout) override;

  void generate() override;

  // inc/dec pending requests
  void incr_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }

  void decr_pending_requests() override {
    const auto old_value =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(old_value, 0) << "pending requests underflow";
  }

  size_t num_pending_requests() {
    return pending_requests_.load(std::memory_order_relaxed);
  }

  std::vector<std::shared_ptr<DiTRequest>> get_running_requests() {
    return running_requests_;
  }

 protected:
  const Options options_;

  // the engine to run the batch
  DiTEngine* engine_;

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<std::shared_ptr<DiTRequest>> request_queue_;

  // a batch of requests in running state
  std::vector<std::shared_ptr<DiTRequest>> running_requests_;

  // response handler
  std::unique_ptr<DiTAsyncResponseProcessor> response_handler_;

  // the number of requests that are waiting to be scheduled
  std::atomic<size_t> pending_requests_{0};

  // build a batch of requests from the priority queue
  virtual std::vector<DiTBatch> prepare_batch();

 private:
  std::vector<DiTBatch> schedule_request(const absl::Duration& timeout);

  // process the batch output
  void process_batch_output();
};

}  // namespace xllm
