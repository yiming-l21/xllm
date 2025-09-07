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

#include <string>
#include <vector>

#include "core/framework/chat_template/jinja_chat_template.h"
#include "mm_data.h"
#include "multimodal.pb.h"

namespace xllm {

struct MMInputItem {
  void clear() {
    type_ = MMType::NONE;
    raw_data_.clear();
  }

  MMType type_ = MMType::NONE;

  std::string raw_data_;  // binary

  torch::Tensor decode_data_;  // image: rgb, [c,h,w], uint8
};

using MMInputItemVec = std::vector<MMInputItem>;
using MMChatMessageVec =
    ::google::protobuf::RepeatedPtrField<proto::MMChatMessage>;
using MMInputDataVec = ::google::protobuf::RepeatedPtrField<proto::MMInputData>;

struct MMInput {
  bool empty() const { return items_.empty(); }

  std::vector<torch::Tensor> get_decode_data(MMType type) const {
    std::vector<torch::Tensor> vec;

    for (const auto& item : items_) {
      if (item.type_ == type) {
        vec.emplace_back(item.decode_data_);
      }
    }
    return std::move(vec);
  }

  MMInputItemVec items_;
};

class MMHandlerSet;
class MMInputHelper {
 public:
  MMInputHelper();
  ~MMInputHelper();

  bool trans(const MMChatMessageVec& vec,
             std::vector<Message>& messages,
             MMInputItemVec& inputs);

 private:
  bool trans(const MMInputDataVec& vec,
             Message::MMContentVec& mmc,
             MMInputItemVec& input);

  std::unique_ptr<MMHandlerSet> mm_handlers_;
};

}  // namespace xllm
