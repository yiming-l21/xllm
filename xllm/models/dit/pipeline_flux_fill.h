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
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "autoencoder_kl.h"
#include "clip_text_model.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/pos_embedding.h"
#include "core/layers/rms_norm.h"
#include "core/layers/rotary_embedding.h"
#include "dit.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "pipeline_flux_utils.h"
#include "t5_encoder.h"
namespace xllm {
// pipeline_flux_fill compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_fill.py

class FluxFillPipelineImpl : public torch::nn::Module {
 public:
  FluxFillPipelineImpl(const DiTModelContext& context) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    latent_channels_ = model_args.latent_channels();

    default_sample_size_ = 128;
    tokenizer_max_length_ = 77;  // TODO: get from config file
    LOG(INFO) << "Initializing FluxFill pipeline...";
    image_processor_ = VAEImageProcessor(
        context.get_model_context("vae"), true, true, false, false, false);
    mask_processor_ = VAEImageProcessor(
        context.get_model_context("vae"), true, false, true, false, true);
    vae_ = VAE(context.get_model_context("vae"));
    LOG(INFO) << "VAE initialized.";
    pos_embed_ = register_module(
        "pos_embed",
        FluxPosEmbed(10000,
                     context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = FluxDiTModel(context.get_model_context("transformer"));
    LOG(INFO) << "DiT transformer initialized.";
    t5_ = T5EncoderModel(context.get_model_context("text_encoder_2"));
    LOG(INFO) << "T5 initialized.";
    clip_text_model_ = CLIPTextModel(context.get_model_context("text_encoder"));
    LOG(INFO) << "CLIP text model initialized.";
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
    LOG(INFO) << "Flux pipeline initialized.";
    register_module("vae", vae_);
    LOG(INFO) << "VAE registered.";
    register_module("vae_image_processor", image_processor_);
    LOG(INFO) << "VAE image processor registered.";
    register_module("mask_processor", mask_processor_);
    LOG(INFO) << "mask processor registered.";
    register_module("transformer", transformer_);
    LOG(INFO) << "DiT transformer registered.";
    register_module("t5", t5_);
    LOG(INFO) << "T5 registered.";
    register_module("scheduler", scheduler_);
    LOG(INFO) << "Scheduler registered.";
    register_module("clip_text_model", clip_text_model_);
    LOG(INFO) << "CLIP text model registered.";
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;
    int64_t height = generation_params.height;
    int64_t width = generation_params.width;
    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);

    auto image = input.images.defined() ? std::make_optional(input.images)
                                        : std::nullopt;
    auto mask_image = input.mask_images.defined()
                          ? std::make_optional(input.mask_images)
                          : std::nullopt;
    auto masked_image_latents =
        input.masked_image_latents.defined()
            ? std::make_optional(input.masked_image_latents)
            : std::nullopt;

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;

    std::vector<torch::Tensor> output =
        forward_(prompts,
                 prompts_2,
                 image,
                 mask_image,
                 masked_image_latents,
                 height,
                 width,
                 generation_params.strength,
                 generation_params.num_inference_steps,
                 std::nullopt,
                 generation_params.guidance_scale,
                 generation_params.num_images_per_prompt,
                 seed,
                 latents,
                 prompt_embeds,
                 pooled_prompt_embeds,
                 generation_params.max_sequence_length);

    DiTForwardOutput out;
    out.tensors = torch::chunk(output[0], output[0].size(0), 0);
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  std::vector<torch::Tensor> forward_(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<torch::Tensor> image = std::nullopt,
      std::optional<torch::Tensor> mask_image = std::nullopt,
      std::optional<torch::Tensor> masked_image_latents = std::nullopt,
      int64_t height = 512,
      int64_t width = 512,
      float strength = 1.0f,
      int64_t num_inference_steps = 50,
      std::optional<std::vector<float>> sigmas = std::nullopt,
      float guidance_scale = 30.0f,
      int64_t num_images_per_prompt = 1,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    guidance_scale_ = guidance_scale;
    torch::Tensor init_image =
        image_processor_->preprocess(image.value(), height, width);

    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }

    torch::Tensor text_ids;
    std::tie(prompt_embeds, pooled_prompt_embeds, text_ids) =
        encode_prompt(prompt,
                      prompt_2,
                      prompt_embeds,
                      pooled_prompt_embeds,
                      num_images_per_prompt,
                      max_sequence_length);

    auto make_default_sigmas = [&](int64_t steps) {
      std::vector<float> result(steps);
      for (int64_t i = 0; i < steps; ++i)
        result[i] = 1.0f - static_cast<float>(i) / steps;
      return result;
    };
    sigmas = sigmas.value_or(make_default_sigmas(num_inference_steps));
    int64_t image_seq_len =
        (height / vae_scale_factor_ / 2) * (width / vae_scale_factor_ / 2);

    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());

    retrieve_timesteps(scheduler_, num_inference_steps, device_, sigmas, mu);
    torch::Tensor timesteps;
    std::tie(timesteps, num_inference_steps) =
        get_timesteps(num_inference_steps, strength);
    CHECK(num_inference_steps >= 1);

    torch::Tensor latent_timestep =
        timesteps.index({torch::indexing::Slice(0, 1)})
            .repeat({batch_size * num_images_per_prompt});

    int64_t num_channels_latents = latent_channels_;
    torch::Tensor latent_image_ids;
    std::tie(latents, latent_image_ids) =
        prepare_latents(init_image,
                        latent_timestep,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        seed,
                        latents);

    if (masked_image_latents.has_value()) {
      masked_image_latents =
          masked_image_latents.value().to(device_).to(dtype_);
    } else {
      mask_image =
          mask_processor_->preprocess(mask_image.value(), height, width);
      torch::Tensor masked_image = init_image * (1 - mask_image.value());

      height = init_image.size(-2);
      width = init_image.size(-1);

      torch::Tensor mask;
      std::tie(mask, masked_image_latents) =
          prepare_mask_latents(mask_image.value(),
                               masked_image,
                               batch_size,
                               num_channels_latents,
                               num_images_per_prompt,
                               height,
                               width,
                               seed);
      masked_image_latents =
          torch::cat({masked_image_latents.value(), mask}, -1);
    }

    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      guidance = torch::full(at::IntArrayRef({1}), guidance_scale, options_);
      guidance = guidance.expand({latents.value().size(0)});
    }

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));

    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(device_);

    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      torch::Tensor t = timesteps[i];
      torch::Tensor timestep = t.expand({latents->size(0)}).to(device_);

      int64_t step_id = i + 1;
      torch::Tensor input_latents =
          torch::cat({latents.value(), masked_image_latents.value()}, 2);

      torch::Tensor noise_pred =
          transformer_->forward(input_latents,
                                prompt_embeds.value(),
                                pooled_prompt_embeds.value(),
                                timestep / 1000,
                                image_rotary_emb,
                                guidance,
                                step_id);
      auto prev_latents = scheduler_->step(noise_pred, t, latents.value());
      latents = prev_latents.detach().to(device_);
    }

    torch::Tensor output_image;
    torch::Tensor unpacked_latents =
        unpack_latents(latents.value(), height, width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;

    output_image = vae_->decode(unpacked_latents);
    output_image = image_processor_->postprocess(output_image, "pil");
    return std::vector<torch::Tensor>{{output_image}};
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxPipeline loading model from" << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto t5_loader = loader->take_component_loader("text_encoder_2");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto tokenizer_2_loader = loader->take_component_loader("tokenizer_2");
    LOG(INFO)
        << "Flux model components loaded, start to load weights to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(device_);
    vae_->load_model(std::move(vae_loader));
    vae_->to(device_);
    t5_->load_model(std::move(t5_loader));
    t5_->to(device_);
    clip_text_model_->load_model(std::move(clip_loader));
    clip_text_model_->to(device_);
    tokenizer_ = tokenizer_loader->tokenizer();
    tokenizer_2_ = tokenizer_2_loader->tokenizer();
  }

 private:
  torch::Tensor get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                     int64_t num_images_per_prompt = 1,
                                     int64_t max_sequence_length = 512) {
    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_2_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      LOG(INFO) << "T5 Original IDs size: " << ids;
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(device_);
    torch::Tensor prompt_embeds = t5_->forward(input_ids);
    prompt_embeds = prompt_embeds.to(device_).to(dtype_);
    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    return prompt_embeds;
  }

  torch::Tensor get_clip_prompt_embeds(std::vector<std::string>& prompt,
                                       int64_t num_images_per_prompt = 1) {
    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      LOG(INFO) << "CLIP Original IDs size: " << ids;
      ids.resize(tokenizer_max_length_, 49407);
      ids.back() = 49407;
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * tokenizer_max_length_);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, tokenizer_max_length_})
            .to(device_);
    auto encoder_output = clip_text_model_->forward(input_ids);
    torch::Tensor prompt_embeds = encoder_output;
    prompt_embeds = prompt_embeds.to(device_).to(dtype_);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, -1});
    return prompt_embeds;
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_mask_latents(
      torch::Tensor mask,
      torch::Tensor masked_image,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t num_images_per_prompt,
      int64_t height,
      int64_t width,
      int64_t seed) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor masked_image_latents;
    if (masked_image.size(1) == num_channels_latents) {
      masked_image_latents = masked_image;
    } else {
      masked_image_latents = vae_->encode(masked_image, seed);
    }

    masked_image_latents =
        (masked_image_latents - vae_shift_factor_) * vae_scaling_factor_;
    masked_image_latents = masked_image_latents.to(device_).to(dtype_);

    batch_size = batch_size * num_images_per_prompt;
    if (mask.size(0) < batch_size) {
      CHECK(batch_size % mask.size(0) == 0)
          << "Masks batch size mismatch: mask cannot be duplicated to match "
             "total batch.";
      mask = mask.repeat({batch_size / mask.size(0), 1, 1, 1});
    }

    if (masked_image_latents.size(0) < batch_size) {
      CHECK(batch_size % masked_image_latents.size(0) == 0)
          << "Masked image batch size mismatch: cannot duplicate to match "
             "total batch.";
      masked_image_latents = masked_image_latents.repeat(
          {batch_size / masked_image_latents.size(0), 1, 1, 1});
    }

    masked_image_latents = pack_latents(
        masked_image_latents, batch_size, num_channels_latents, height, width);

    mask = mask.select(1, 0);
    mask = mask.view(
        {batch_size, height, vae_scale_factor_, width, vae_scale_factor_});
    mask = mask.permute({0, 2, 4, 1, 3});
    mask = mask.reshape(
        {batch_size, vae_scale_factor_ * vae_scale_factor_, height, width});
    mask = pack_latents(
        mask, batch_size, vae_scale_factor_ * vae_scale_factor_, height, width);
    mask = mask.to(device_).to(dtype_);

    return {mask, masked_image_latents};
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> prompt_2,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> pooled_prompt_embeds,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      std::vector<std::string> prompt_2_list;
      if (prompt_2.has_value()) {
        prompt_2_list = prompt_2.value();
      }
      if (prompt_2_list.empty()) {
        prompt_2_list = prompt_list;
      }
      pooled_prompt_embeds =
          get_clip_prompt_embeds(prompt_list, num_images_per_prompt);
      prompt_embeds = get_t5_prompt_embeds(
          prompt_2_list, num_images_per_prompt, max_sequence_length);
    }
    torch::Tensor text_ids = torch::zeros({prompt_embeds.value().size(1), 3},
                                          torch::device(device_).dtype(dtype_));

    return std::make_tuple(prompt_embeds.value(),
                           pooled_prompt_embeds.has_value()
                               ? pooled_prompt_embeds.value()
                               : torch::Tensor(),
                           text_ids);
  }

  torch::Tensor encode_vae_image(const torch::Tensor& image, int64_t seed) {
    torch::Tensor latents = vae_->encode(image, seed);
    latents = (latents - vae_shift_factor_) * vae_scaling_factor_;
    return latents;
  }

  std::pair<torch::Tensor, int64_t> get_timesteps(int64_t num_inference_steps,
                                                  double strength) {
    int64_t init_timestep =
        std::min(static_cast<int64_t>(num_inference_steps * strength),
                 num_inference_steps);

    int64_t t_start = std::max(num_inference_steps - init_timestep, int64_t(0));
    int64_t start_idx = t_start * scheduler_->order();
    auto timesteps =
        scheduler_->timesteps().slice(0, start_idx).to(device_).to(dtype_);
    scheduler_->set_begin_index(start_idx);
    return {timesteps, num_inference_steps - t_start};
  }

  torch::Tensor prepare_latent_image_ids(int64_t batch_size,
                                         int64_t height,
                                         int64_t width) {
    torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options_);
    torch::Tensor height_range = torch::arange(height, options_).unsqueeze(1);
    latent_image_ids.select(2, 1) += height_range;
    torch::Tensor width_range = torch::arange(width, options_).unsqueeze(0);
    latent_image_ids.select(2, 2) += width_range;
    latent_image_ids = latent_image_ids.view({height * width, 3});
    return latent_image_ids;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents,
                             int64_t batch_size,
                             int64_t num_channels_latents,
                             int64_t height,
                             int64_t width) {
    torch::Tensor latents_packed = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents_packed = latents_packed.permute({0, 2, 4, 1, 3, 5});
    latents_packed = latents_packed.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return latents_packed;
  }

  torch::Tensor unpack_latents(const torch::Tensor& latents,
                               int64_t height,
                               int64_t width,
                               int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t num_patches = latents.size(1);
    int64_t channels = latents.size(2);
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor latents_unpacked =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
    latents_unpacked = latents_unpacked.reshape(
        {batch_size, channels / (2 * 2), height, width});

    return latents_unpacked;
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      torch::Tensor image,
      torch::Tensor timesteps,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, height, width};
    torch::Tensor latent_image_ids =
        prepare_latent_image_ids(batch_size, height / 2, width / 2);
    if (latents.has_value()) {
      return {latents.value().to(device_).to(dtype_), latent_image_ids};
    }

    torch::Tensor image_latents;
    if (image.size(1) != latent_channels_) {
      image_latents = encode_vae_image(image, seed);
    } else {
      image_latents = image;
    }
    int64_t additional_image_per_prompt;
    if (batch_size > image_latents.size(0) &&
        batch_size % image_latents.size(0) == 0) {
      additional_image_per_prompt = batch_size / image_latents.size(0);
      image_latents =
          image_latents.repeat({additional_image_per_prompt, 1, 1, 1});
    } else if (batch_size > image_latents.size(0) &&
               batch_size % image_latents.size(0) != 0) {
      LOG(FATAL) << "Cannot match batch_size with input images.";
    } else {
      image_latents = torch::cat({image_latents}, 0);
    }
    auto noise = randn_tensor(shape, seed, options_);
    latents = scheduler_->scale_noise(image_latents, timesteps, noise);
    latents = pack_latents(
        latents.value(), batch_size, num_channels_latents, height, width);
    return {latents.value(), latent_image_ids};
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor image_processor_{nullptr};
  VAEImageProcessor mask_processor_{nullptr};
  FluxDiTModel transformer_{nullptr};
  T5EncoderModel t5_{nullptr};
  CLIPTextModel clip_text_model_{nullptr};
  int vae_scale_factor_;
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int tokenizer_max_length_;
  int default_sample_size_;
  float guidance_scale_ = 1.0f;
  int64_t latent_channels_;
  torch::TensorOptions options_;
  FluxPosEmbed pos_embed_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_2_;

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
};

TORCH_MODULE(FluxFillPipeline);

REGISTER_DIT_MODEL(fluxfill, FluxFillPipeline);
}  // namespace xllm
