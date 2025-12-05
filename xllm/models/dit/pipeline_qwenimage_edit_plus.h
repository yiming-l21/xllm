#pragma once
#include "core/layers/pos_embedding.h"
#include "core/layers/rotary_embedding.h"
#include "pipeline_qwenimage_base.h"
#define CONDITION_IMAGE_SIZE 147456
#define VAE_IMAGE_SIZE 1048576
#include "core/framework/state_dict/state_dict.h"

namespace xllm {
namespace qwenimage {
class QwenImageEditPlusPipelineImpl : public QwenImagePipelineBaseImpl {
 public:
  QwenImageEditPlusPipelineImpl(const DiTModelContext& context)
      : vae_model_args_(context.get_model_args("vae")) {
    options_ = context.get_tensor_options();
    dtype_ = options_.dtype().toScalarType();
    device_ = options_.device();
    LOG(INFO) << "model info " << dtype_ << " ; " << options_.device();

    in_channels_ = context.get_model_args("transformer").in_channels();

    vae_scale_factor_ = static_cast<int>(
        std::pow(2, vae_model_args_.temperal_downsample().size()));
    latent_channels_ = vae_model_args_.z_dim();
    tokenizer_max_length_ = 1024;

    prompt_template_encode_ =
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how "
        "the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while "
        "maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        "assistant\n";
    prompt_template_encode_start_idx_ = 64;
    default_sample_size_ = 128;

    vae_ = AutoencoderKLQwenImage(context.get_model_context("vae"));
    transformer_ =
        QwenImageTransformer2DModel(context.get_model_context("transformer"));
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));

    vae_image_processor_ = VAEImageProcessor(context.get_model_context("vae"),
                                             true,
                                             true,
                                             false,
                                             false,
                                             false,
                                             latent_channels_);
    register_module("vae", vae_);
    register_module("scheduler", scheduler_);
    register_module("transformer", transformer_);
    register_module("vae_image_processor", vae_image_processor_);
  }

  std::vector<torch::Tensor> _extract_masked_hidden(
      const torch::Tensor& hidden_states,
      const torch::Tensor& mask) {
    auto bool_mask = mask.to(torch::kBool);
    auto valid_lengths = bool_mask.sum(1);

    auto valid_lengths_cpu = valid_lengths.to(torch::kCPU).contiguous();

    std::vector<int64_t> lengths_list;
    lengths_list.reserve(valid_lengths_cpu.numel());

    int64_t* lengths_ptr = valid_lengths_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < valid_lengths_cpu.numel(); ++i) {
      lengths_list.push_back(lengths_ptr[i]);
    }

    auto selected = hidden_states.index({bool_mask});
    auto split_result = torch::split(selected, lengths_list, 0);

    return std::vector<torch::Tensor>(split_result.begin(), split_result.end());
  }
  /*
  std::pair<torch::Tensor, torch::Tensor> get_qwen_prompt_embeds(
        const std::vector<std::string>& prompt,
        const std::vector<torch::Tensor>& image,
        torch::TensorOptions options) {

        std::string img_prompt_template = "Picture {}:
  <|vision_start|><|image_pad|><|vision_end|>"; std::string base_img_prompt =
  "";

        if (!image.empty()) {
            if (image.size() > 1) {
                for (size_t i = 0; i < image.size(); ++i) {
                    base_img_prompt += fmt::format(img_prompt_template, i + 1);
                }
            } else {
                base_img_prompt = fmt::format(img_prompt_template, 1);
            }
        }

        // 处理文本提示
        std::vector<std::string> txt;
        for (const auto& p : prompt) {
            txt.push_back(fmt::format(prompt_template_encode_, base_img_prompt +
  p));
        }
        ModelInputParams input_params = ModelInputParams();

  }
  */
  torch::Tensor retrieve_latents(const AutoencoderKLOutput& encoder_output,
                                 int64_t seed = 42,
                                 const std::string& sample_mode = "sample") {
    if (sample_mode == "sample") {
      return encoder_output.latent_dist.sample(seed);
    } else if (sample_mode == "argmax") {
      return encoder_output.latent_dist.mode();
    } else {
      CHECK(false)
          << "sample_mode is expected to be 'sample' or 'argmax', but get: "
          << sample_mode;
      return torch::Tensor();
    }
  }

  torch::Tensor _pack_latents(torch::Tensor latents,
                              int64_t batch_size,
                              int64_t num_channels_latents,
                              int64_t height,
                              int64_t width) {
    latents = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents = latents.permute({0, 2, 4, 1, 3, 5});
    latents = latents.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return latents;
  }

  torch::Tensor _unpack_latents(torch::Tensor latents,
                                int64_t height,
                                int64_t width,
                                int64_t vae_scale_factor) {
    auto sizes = latents.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_patches = sizes[1];
    int64_t channels = sizes[2];

    height = 2 * (height / (vae_scale_factor * 2));
    width = 2 * (width / (vae_scale_factor * 2));

    latents =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents = latents.permute({0, 3, 1, 4, 2, 5});
    latents =
        latents.reshape({batch_size, channels / (2 * 2), 1, height, width});

    return latents;
  }

  torch::Tensor _encode_vae_image(torch::Tensor image,
                                  int64_t seed,
                                  torch::Device device) {
    auto tensor_dict = StateDictFromSafeTensor::load(
        "/export/home/shanchenfeng/xllm_build/xllm_qwenimage/xllm/xllm/"
        "tensor_dict.safetensors");
    bool weight_loaded = false;
    weight::load_weight(*tensor_dict, "image", image, weight_loaded);
    auto image_latents = retrieve_latents(vae_->encode(image), seed, "argmax");
    // torch::save(image, "qwen_image_raw.pt");
    // torch::save(image_latents, "qwen_image_raw_latents.pt");
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = latents_std.view({1, latent_channels_, 1, 1, 1})
                      .to(device, image_latents.dtype());
    LOG(INFO) << "before latents";
    image_latents = (image_latents - latents_mean) / latents_std;
    LOG(INFO) << "after latents";
    // torch::save(image_latents, "qwen_image_raw_latents.pt");
    return image_latents;
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      const std::vector<torch::Tensor>& images,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      torch::TensorOptions& options,
      int64_t seed,
      torch::Tensor latents = torch::Tensor()) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    std::vector<int64_t> shape = {
        batch_size, 1, num_channels_latents, height, width};

    torch::Tensor image_latents;
    if (!images.empty()) {
      std::vector<torch::Tensor> all_image_latents;

      for (const auto& image : images) {
        auto current_image = image.to(options);
        torch::Tensor current_image_latents;
        LOG(INFO) << "begin vae inference";
        for (auto size : image.sizes()) {
          LOG(INFO) << "size : " << size;
        }

        if (current_image.size(1) != latent_channels_) {
          current_image_latents =
              _encode_vae_image(current_image, seed, device_);
        } else {
          current_image_latents = current_image;
        }

        current_image_latents = torch::cat({current_image_latents}, 0);
        for (auto size : current_image_latents.sizes()) {
          LOG(INFO) << size;
        }
        int64_t image_latent_height = current_image_latents.size(3);
        int64_t image_latent_width = current_image_latents.size(4);

        current_image_latents = _pack_latents(current_image_latents,
                                              batch_size,
                                              num_channels_latents,
                                              image_latent_height,
                                              image_latent_width);
        all_image_latents.emplace_back(current_image_latents);
      }

      image_latents = torch::cat(all_image_latents, 1);
    }

    if (!latents.defined()) {
      latents = randn_tensor(shape, seed, options);
      latents = _pack_latents(
          latents, batch_size, num_channels_latents, height, width);
      auto tensor_dict = StateDictFromSafeTensor::load(
          "/export/home/shanchenfeng/xllm_build/xllm_qwenimage/xllm/xllm/"
          "latents2.safetensors");
      bool weight_loaded = false;
      weight::load_weight(*tensor_dict, "latents2", latents, weight_loaded);
      // torch::save(latents, "latents.pt");
    } else {
      latents = latents.to(options);
    }
    LOG(INFO) << "outside latents";
    return std::make_pair(latents, image_latents);
  }
  // torch::Tensor forward(torch::Tensor input){return torch::empty({0});}
  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& generation_params = input.generation_params;
    auto height = generation_params.height;
    auto width = generation_params.width;
    auto true_cfg_scale = generation_params.true_cfg_scale;
    auto num_inference_steps = generation_params.num_inference_steps;
    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = input.prompts;
    auto prompts_2 = input.prompts_2;
    auto negative_prompts = input.negative_prompts;
    auto negative_prompts_2 = input.negative_prompts_2;

    auto latents = input.latents;
    auto prompt_embeds = input.prompt_embeds;
    auto negative_prompt_embeds = input.negative_prompt_embeds;
    auto pooled_prompt_embeds = input.pooled_prompt_embeds;
    auto negative_pooled_prompt_embeds = input.negative_pooled_prompt_embeds;
    auto images = input.images.to(options_.device(), dtype_);
    double height_size = images.size(2);
    double width_size = images.size(3);
    int64_t num_images_per_prompt = 1;

    double aspect_ratio = height_size / width_size;
    auto [calculated_width, calculated_height] =
        calculate_dimensions(1024 * 1024, aspect_ratio);

    height = (height == 0) ? calculated_height : height;
    width = (width == 0) ? calculated_width : width;

    int multiple_of = vae_scale_factor_ * 2;
    width = (width / multiple_of) * multiple_of;
    height = (height / multiple_of) * multiple_of;

    current_timestep_ = torch::Tensor();

    // 2. 定义调用参数
    int64_t batch_size = prompts.size();

    // 3. 预处理图像
    std::vector<torch::Tensor> condition_images;
    std::vector<torch::Tensor> vae_images;
    std::vector<std::pair<int64_t, int64_t>> condition_image_sizes;
    std::vector<std::pair<int64_t, int64_t>> vae_image_sizes;
    // torch::save(images, "ori_image.pt");
    if (images.defined() && !(images.size(1) == latent_channels_)) {
      for (size_t i = 0; i < images.size(0); i++) {
        auto [condition_width, condition_height] =
            calculate_dimensions(CONDITION_IMAGE_SIZE, aspect_ratio);
        auto [vae_width, vae_height] =
            calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio);
        LOG(INFO) << "aspect_ratio : " << aspect_ratio;
        LOG(INFO) << vae_width << " : " << vae_height;

        condition_image_sizes.push_back({condition_width, condition_height});
        vae_image_sizes.push_back({vae_width, vae_height});

        auto img = images[i].unsqueeze(0);
        for (auto size : img.sizes()) {
          LOG(INFO) << "img size" << size;
        }
        auto condition_img = vae_image_processor_->resize(
            img, condition_height, condition_width);
        auto vae_img =
            vae_image_processor_->preprocess(img, vae_height, vae_width)
                .unsqueeze(2);
        for (auto size : vae_img.sizes()) {
          LOG(INFO) << "vae img size" << size;
        }
        condition_images.push_back(condition_img);
        vae_images.push_back(vae_img);
      }
    }

    // 检查是否有负提示
    bool has_neg_prompt = negative_prompts.size() > 0;

    bool do_true_cfg = (true_cfg_scale > 1.0) && has_neg_prompt;
    /*
    // 编码提示
    auto [new_prompt_embeds, new_prompt_embeds_mask] = encode_prompt(
        condition_images, prompt, prompt_embeds, prompt_embeds_mask,
        device_, num_images_per_prompt, max_sequence_length
    );

    torch::Tensor new_negative_prompt_embeds, new_negative_prompt_embeds_mask;
    if (do_true_cfg) {
        std::tie(final_negative_prompt_embeds,
    final_negative_prompt_embeds_mask) = encode_prompt( condition_images,
    negative_prompt, negative_prompt_embeds, negative_prompt_embeds_mask,
            device_, num_images_per_prompt, max_sequence_length
        );
    }
    */
    auto final_prompt_embeds = torch::ones({1, 230, 3584}, options_);
    auto final_prompt_embeds_mask = torch::ones({1, 230}, options_);
    auto final_negative_prompt_embeds = torch::ones({1, 209, 3584}, options_);
    auto final_negative_prompt_embeds_mask = torch::ones({1, 209}, options_);

    auto tensor_dict = StateDictFromSafeTensor::load(
        "/export/home/shanchenfeng/xllm_build/xllm_qwenimage/xllm/xllm/models/"
        "dit/model.safetensors");
    bool weight_loaded = false;
    bool weight_loaded1 = false;
    bool weight_loaded2 = false;
    bool weight_loaded3 = false;
    weight::load_weight(
        *tensor_dict, "prompt_embeds", final_prompt_embeds, weight_loaded);
    weight::load_weight(*tensor_dict,
                        "prompt_embeds_mask",
                        final_prompt_embeds_mask,
                        weight_loaded1);
    weight::load_weight(*tensor_dict,
                        "negative_prompt_embeds",
                        final_negative_prompt_embeds,
                        weight_loaded2);
    weight::load_weight(*tensor_dict,
                        "negative_prompt_embeds_mask",
                        final_negative_prompt_embeds_mask,
                        weight_loaded3);
    int64_t num_channels_latents = in_channels_ / 4;
    torch::Tensor final_latents;
    torch::Tensor image_latents;

    std::tie(final_latents, image_latents) =
        prepare_latents(vae_images,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        options_,
                        42,
                        latents);

    // std::vector<std::vector<std::vector<int64_t>>> img_shapes;
    std::vector<std::vector<int64_t>> main_shape = {
        {1, height / vae_scale_factor_ / 2, width / vae_scale_factor_ / 2}};

    for (const auto& [vae_width, vae_height] : vae_image_sizes) {
      main_shape.push_back({1,
                            vae_height / vae_scale_factor_ / 2,
                            vae_width / vae_scale_factor_ / 2});
    }
    /*
    for (int i = 0; i < batch_size; ++i) {
        img_shapes.push_back(main_shape);
    }
    */

    std::vector<float> new_sigmas;
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                      (num_inference_steps - 1) *
                                      (1.0f - 1.0f / num_inference_steps));
      LOG(INFO) << "sigma " << new_sigmas[i];
    }

    int64_t image_seq_len = final_latents.size(1);
    LOG(INFO) << "image_seq_len" << image_seq_len;
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());
    LOG(INFO) << "mu is" << mu;
    auto [timesteps, num_inference_steps_actual] =
        xllm::qwenimage::retrieve_timesteps(
            scheduler_, num_inference_steps, device_, new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));
    // torch::save(timesteps, "timesteps.pt");
    num_timesteps_ = timesteps.size(0);
    LOG(INFO) << "after retrive timesteps";
    torch::Tensor txt_seq_lens;
    if (final_prompt_embeds_mask.defined()) {
      txt_seq_lens = final_prompt_embeds_mask.sum(1);
    }
    torch::Tensor negative_txt_seq_lens;
    if (do_true_cfg && final_negative_prompt_embeds_mask.defined()) {
      negative_txt_seq_lens = final_negative_prompt_embeds_mask.sum(1);
    }

    scheduler_->set_begin_index(0);
    LOG(INFO) << "after set begin index";
    for (int i = 0; i < timesteps.size(0); ++i) {
      auto t = timesteps[i];
      current_timestep_ = t;

      auto latent_model_input = final_latents;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({final_latents, image_latents}, 1);
      }

      // 扩展时间步长到批次维度
      auto timestep_expanded =
          t.expand({final_latents.size(0)}).to(final_latents.dtype());
      LOG(INFO) << "begine forwad";
      torch::Tensor noise_pred;
      {
        noise_pred = transformer_->forward(latent_model_input,
                                           final_prompt_embeds,
                                           final_prompt_embeds_mask,
                                           timestep_expanded / 1000.0,
                                           main_shape,
                                           txt_seq_lens);
        noise_pred = noise_pred.slice(1, 0, final_latents.size(1));
        // torch::save(noise_pred, "noise_pred0.pt");
      }

      if (do_true_cfg) {
        torch::Tensor neg_noise_pred;
        {
          // 无条件预测
          neg_noise_pred =
              transformer_->forward(latent_model_input,
                                    final_negative_prompt_embeds,
                                    final_negative_prompt_embeds_mask,
                                    timestep_expanded / 1000.0,
                                    main_shape,
                                    negative_txt_seq_lens);
          // torch::save(latent_model_input, "latent_model_input.pt");
          // torch::save(final_negative_prompt_embeds,
          // "final_negative_prompt_embeds.pt");

          neg_noise_pred = neg_noise_pred.slice(1, 0, final_latents.size(1));
          // torch::save(neg_noise_pred, "neg_noise_pred.pt");
        }

        auto comb_pred =
            neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred);
        // torch::save(comb_pred, "comb_pred.pt");
        auto cond_norm = torch::norm(noise_pred, 2, -1, true);
        // torch::save(cond_norm, "cond_norm.pt");
        auto noise_norm = torch::norm(comb_pred, 2, -1, true);
        // torch::save(noise_norm, "noise_norm.pt");
        noise_pred = comb_pred * (cond_norm / noise_norm);
        // torch::save(noise_pred, "final_noise_pred.pt");
      }

      auto latents_dtype = final_latents.dtype();
      final_latents = scheduler_->step(noise_pred, t, final_latents);
      torch::save(final_latents, "final_latents_0.pt");
      // std::exit(0);
      if (final_latents.dtype() != latents_dtype) {
        final_latents = final_latents.to(latents_dtype);
      }
    }

    current_timestep_ = torch::Tensor();

    // 7. 后处理
    torch::Tensor output_image;

    auto unpacked_latents =
        _unpack_latents(final_latents, height, width, vae_scale_factor_)
            .to(dtype_);
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device_, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = 1.0 / latents_std.view({1, latent_channels_, 1, 1, 1})
                            .to(device_, image_latents.dtype());

    unpacked_latents = unpacked_latents / latents_std + latents_mean;
    torch::save(unpacked_latents, "unpacked_latents.pt");
    output_image = vae_->decode(unpacked_latents).sample.squeeze(2);
    for (auto size : output_image.sizes()) {
      LOG(INFO) << "output size" << size;
    }
    torch::save(output_image, "output_image_before.pt");
    output_image = vae_image_processor_->postprocess(output_image, "pil");
    for (auto size : output_image.sizes()) {
      LOG(INFO) << "output size" << size;
    }
    torch::save(output_image, "output_image.pt");
    auto output = std::vector<torch::Tensor>{{output_image}};
    DiTForwardOutput out;
    out.tensors = output;  // torch::chunk(output[0], output[0].size(0), 0);
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "QwenImageEditPlusPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto processor_loader = loader->take_component_loader("processor");
    LOG(INFO) << " QwenImageEditplus model components loaded, start to load "
                 "weights to sub models";
    /*
    qwen_image_processor_ =
    std::make_unique<Qwen2VLImageProcessor>(processor_loader->model_args());
    qwen_tokenizer_ = processor_loader->tokenizer();
    qwen_processor_ =
    std::make_unique<Qwen2VLProcessor>(std::move(qwen_image_processor_),
                                                         std::move(qwen_tokenizer_),
                                                         processor_loader->model_args()
                                                         );
    tokenizer_ = tokenizer_loader->tokenizer();
    */
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device(), dtype_);
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device(), dtype_);
    /*
    tokenizer_ = tokenizer_loader->tokenizer();
    transformer_->to(device_);
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device(), dtype_);
    t5_->load_model(std::move(t5_loader));
    t5_->to();
    clip_text_model_->load_model(std::move(clip_loader));
    clip_text_model_->to(device_);
    tokenizer_ = tokenizer_loader->tokenizer();
    tokenizer_2_ = tokenizer_2_loader->tokenizer();
    */
  }

 private:
  int64_t vae_scale_factor_;
  int64_t latent_channels_;
  int64_t tokenizer_max_length_;
  int64_t prompt_template_encode_start_idx_;
  int64_t default_sample_size_;
  int64_t in_channels_;
  int64_t num_timesteps_;
  torch::Tensor current_timestep_;
  string prompt_template_encode_;
  const ModelArgs& vae_model_args_;
};

REGISTER_MODEL_ARGS(Qwen2Tokenizer, [&] {});
TORCH_MODULE(QwenImageEditPlusPipeline);

REGISTER_DIT_MODEL(qwen_image_edit_plus, QwenImageEditPlusPipeline);
}  // namespace qwenimage
}  // namespace xllm
