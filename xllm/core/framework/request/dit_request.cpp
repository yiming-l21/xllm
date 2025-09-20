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

#include "dit_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "api_service/call.h"

namespace {
class OpenCVImageEncoder {
 public:
  // t float32, cpu, chw
  bool encode(const torch::Tensor& t, std::string& raw_data) {
    if (!valid(t)) {
      return false;
    }

    auto img = t.permute({1, 2, 0}).contiguous();
    cv::Mat mat(img.size(0), img.size(1), CV_32FC3, img.data_ptr<float>());

    cv::Mat mat_8u;
    mat.convertTo(mat_8u, CV_8UC3, 255.0);

    // rgb -> bgr
    cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);

    std::vector<uchar> data;
    if (!cv::imencode(".png", mat_8u, data)) {
      LOG(ERROR) << "image encode faild";
      return false;
    }

    raw_data.assign(data.begin(), data.end());
    return true;
  }

 private:
  bool valid(const torch::Tensor& t) {
    if (t.dim() != 3 || t.size(0) != 3) {
      LOG(ERROR) << "input tensor must be 3HW  tensor";
      return false;
    }

    if (t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) {
      LOG(ERROR) << "tensor must be cpu float32";
      return false;
    }

    return true;
  }
};
}  // namespace

namespace xllm {
DiTRequest::DiTRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DiTRequestState& state,
                       const std::string& service_request_id)
    : RequestBase(request_id, x_request_id, x_request_time, service_request_id),
      state_(state) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

int DiTRequest::handle_forward_output(int offset,
                                      const DiTForwardOutput& output) {
  int count = state_.generation_params().num_images_per_prompt.value();
  output_ = output.slice(offset, count);

  return count;
}

const DiTRequestOutput DiTRequest::generate_output() {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = false;

  DiTGenerationOutput result;
  result.height = state_.generation_params().height;
  result.width = state_.generation_params().width;
  result.seed = state_.generation_params().seed.value();

  OpenCVImageEncoder encoder;
  int count = state_.generation_params().num_images_per_prompt.value();
  for (int idx = 0; idx < count; ++idx) {
    torch::Tensor image = output_.tensor.slice(0, idx, idx + 1)
                              .squeeze(0)
                              .cpu()
                              .to(torch::kFloat32)
                              .contiguous();
    encoder.encode(image, result.image);
    output.outputs.push_back(result);
  }

  return output;
}

}  // namespace xllm
