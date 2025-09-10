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

#include "flux_model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "core/util/json_reader.h"
#include "model_loader.h"

namespace xllm {

DITModelLoader::DITModelLoader(const std::string& model_root_path)
    : model_root_path_(model_root_path) {
  if (!std::filesystem::exists(model_root_path_)) {
    LOG(ERROR) << "Model root path does not exist: " << model_root_path_;
  }

  // check if model_index.json exists
  std::filesystem::path root_path(model_root_path_);
  std::filesystem::path index_file = root_path / "model_index.json";
  const std::string model_index_file = index_file.string();
  if (!std::filesystem::exists(model_index_file)) {
    LOG(ERROR) << "Model index file does not exist: " << model_index_file;
  }

  JsonReader model_index_reader;
  if (!model_index_reader.parse(model_index_file)) {
    LOG(ERROR) << "Failed to parse model index file: " << model_index_file;
  }

  LOG(INFO) << "Success to parse model index file: " << model_index_file;

  const nlohmann::json root_json = model_index_reader.data();
  if (!root_json.is_object()) {
    LOG(FATAL) << "DITModelLoader: model_index.json root is not an object!";
  }

  // parse model_index.json & initialize model_loader
  for (const auto& [json_key, json_value] : root_json.items()) {
    if (!json_value.is_array() || json_value.size() != 2) {
      LOG(WARNING) << "DITModelLoader: Invalid format for component! "
                   << "JsonKey=" << json_key
                   << ", Expected [library, class_name] array";
      continue;
    }

    const std::string component_name = json_value[1].get<std::string>();
    std::filesystem::path component_folder_path =
        std::filesystem::path(model_root_path_) / json_key;
    const std::string component_folder = component_folder_path.string();
    if (!std::filesystem::exists(component_folder)) {
      LOG(WARNING) << "DITModelLoader: Component folder not found! "
                   << "ComponentName=" << component_name
                   << ", Folder=" << component_folder;
      continue;
    }
    if (!std::filesystem::is_directory(component_folder)) {
      LOG(WARNING) << "DITModelLoader: Component path is not a directory! "
                   << "ComponentName=" << component_name
                   << ", Path=" << component_folder;
      continue;
    }

    // create model loader for each subfolder
    std::unique_ptr<ModelLoader> sub_loader =
        ModelLoader::create(component_folder);
    if (!sub_loader) {
      LOG(WARNING) << "Failed to create loader for: " << component_name;
      continue;
    }

    name_to_folder_[component_name] = component_folder;
    name_to_loader_[component_name] = std::move(sub_loader);
    sub_model_loaders_.push_back(name_to_loader_[component_name].get());
  }
}

const ModelLoader* DITModelLoader::get_sub_model_loader_by_name(
    const std::string& component_name) const {
  auto it = name_to_loader_.find(component_name);
  if (it == name_to_loader_.end()) {
    LOG(WARNING) << "Component not found: " << component_name;
    return nullptr;
  }
  return it->second.get();
}

const ModelLoader* DITModelLoader::get_sub_model_loader_by_folder(
    const std::string& component_folder) const {
  std::filesystem::path abs_folder =
      std::filesystem::absolute(component_folder);
  for (const auto& [name, folder] : name_to_folder_) {
    if (folder == abs_folder.string()) {
      return name_to_loader_.at(name).get();
    }
  }
  LOG(WARNING) << "Component folder not found: " << component_folder;
  return nullptr;
}

const std::vector<ModelLoader*>& DITModelLoader::get_all_sub_model_loaders()
    const {
  return sub_model_loaders_;
}

std::vector<std::string> DITModelLoader::get_all_sub_model_names() const {
  std::vector<std::string> names;
  for (const auto& [name, _] : name_to_loader_) {
    names.push_back(name);
  }
  return names;
}

bool DITModelLoader::has_sub_model(const std::string& component_name) const {
  return name_to_loader_.count(component_name) > 0;
}
}  // namespace xllm