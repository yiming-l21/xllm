/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "dit_request_params.h"

namespace xllm {

struct DITRequestState {
 public:
  DITRequestState(InputParams&& input_params,
                  GenerationParams&& generation_params)
      : input_params_(std::move(input_params)),
        generation_params_(std::move(generation_params)) {}
  DITRequestState() {}
  InputParams& input_params() { return input_params_; }
  GenerationParams& generation_params() { return generation_params_; }

 private:
  InputParams input_params_;
  GenerationParams generation_params_;
};

class DITRequest : public RequestBase {
 public:
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const DiTRequestState& state,
          const std::string& service_request_id = "");

  DITRequestState& state() { return state_; }
private:
  DITRequestState state_;
};

}  // namespace xllm
