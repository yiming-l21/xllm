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

#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "model_loader.h"
namespace xllm {
class DITModelLoader {
 public:
  explicit DITModelLoader(const std::string& model_root_path);

  const ModelLoader* get_sub_model_loader_by_name(
      const std::string& component_name) const;

  const ModelLoader* get_sub_model_loader_by_folder(
      const std::string& component_folder) const;

  const std::vector<ModelLoader*>& get_all_sub_model_loaders() const;

  std::vector<std::string> get_all_sub_model_names() const;
  ;

  bool has_sub_model(const std::string& component_name) const;
  ;

 private:
  std::string model_root_path_;

  std::unordered_map<std::string, std::unique_ptr<ModelLoader>> name_to_loader_;

  std::unordered_map<std::string, std::string> name_to_folder_;

  std::vector<ModelLoader*> sub_model_loaders_;
};
}  // namespace xllm