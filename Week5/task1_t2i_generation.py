from diffusers import (
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    AuraFlowPipeline,
    FluxPipeline,
    Flux2Pipeline,
    AutoModel,
)
from transformers import Mistral3ForConditionalGeneration
import argparse
import torch

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Model registry: short name -> HuggingFace ID, pipeline class, inference params
MODEL_CONFIGS = {
    "sd1.4": {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "pipeline": StableDiffusionPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 7.5},
        "negative_prompt": True,
    },
    "sd1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": StableDiffusionPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 7.5},
        "negative_prompt": True,
    },
    "sd1.5-lcm": {
        "model_id": "SimianLuo/LCM_Dreamshaper_v7",
        "pipeline": StableDiffusionPipeline,
        "kwargs": {"num_inference_steps": 4, "guidance_scale": 8.0},
        "negative_prompt": True,
    },
    "sd2.1": {
        "model_id": "sd2-community/stable-diffusion-2-1",
        "pipeline": StableDiffusionPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 7.5},
        "negative_prompt": True,
    },
    "sd2.1-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "pipeline": AutoPipelineForText2Image,
        "variant": "fp16",
        "kwargs": {"num_inference_steps": 1, "guidance_scale": 0.0},
        "negative_prompt": True,
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline": AutoPipelineForText2Image,
        "variant": "fp16",
        "kwargs": {"num_inference_steps": 1, "guidance_scale": 0.0},
        "negative_prompt": True,
    },
    "sd3.5-medium": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline": StableDiffusion3Pipeline,
        "kwargs": {"num_inference_steps": 40, "guidance_scale": 4.5},
        "negative_prompt": True,
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "refiner_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "pipeline": StableDiffusionXLPipeline,
        "refiner_pipeline": StableDiffusionXLImg2ImgPipeline,
        "kwargs": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent"},
        "refiner_kwargs": {"num_inference_steps": 40, "denoising_start": 0.8},
        "negative_prompt": True,
    },
    "playground-v2.5": {
        "model_id": "playgroundai/playground-v2.5-1024px-aesthetic",
        "pipeline": DiffusionPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 3.0},
        "negative_prompt": True,
    },
    "auraflow-v0.2": {
        "model_id": "fal/AuraFlow-v0.2",
        "pipeline": AuraFlowPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 3.5},
        "negative_prompt": False,
    },
    "sd3.5-large": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": StableDiffusion3Pipeline,
        "kwargs": {"num_inference_steps": 40, "guidance_scale": 4.5},
        "negative_prompt": True,
    },
    "sd3.5-large-turbo": {
        "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "pipeline": StableDiffusion3Pipeline,
        "kwargs": {"num_inference_steps": 4, "guidance_scale": 1.0},
        "negative_prompt": True,
    },
    "flux-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "pipeline": FluxPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 3.5},
        "negative_prompt": False,
    },
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": FluxPipeline,
        "kwargs": {"num_inference_steps": 4, "guidance_scale": 0.0},
        "negative_prompt": False,
    },
    "flux2-dev": {
        "model_id": "black-forest-labs/FLUX.2-dev",
        "pipeline": DiffusionPipeline,
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 3.5},
        "negative_prompt": False,
    },
    "flux2-dev-4bit": {
        "model_id": "diffusers/FLUX.2-dev-bnb-4bit",
        "kwargs": {"num_inference_steps": 50, "guidance_scale": 4.0},
        "negative_prompt": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running inference on various models"
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Models to run. Available: {', '.join(MODEL_CONFIGS.keys())}",
    )
    return parser.parse_args()


def run_sdxl(config, prompt, save_dir, negative_prompt=None):
    base = config["pipeline"].from_pretrained(
        config["model_id"], torch_dtype=torch.float16
    ).to("cuda")
    refiner = config["refiner_pipeline"].from_pretrained(
        config["refiner_id"],
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
    ).to("cuda")

    neg = negative_prompt if config.get("negative_prompt") else None
    latent = base(prompt=prompt, negative_prompt=neg, **config["kwargs"]).images
    image = refiner(prompt=prompt, negative_prompt=neg, image=latent, **config["refiner_kwargs"]).images[0]
    slug = prompt[:40].replace(" ", "_").replace("/", "-")
    image.save(f"{save_dir}/{slug}_sdxl.png")


def run_flux2_4bit(name, config, prompt, save_dir, negative_prompt=None):
    repo_id = config["model_id"]
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    transformer = AutoModel.from_pretrained(
        repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, text_encoder=text_encoder, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    image = pipe(prompt=prompt, **config["kwargs"]).images[0]
    slug = prompt[:40].replace(" ", "_").replace("/", "-")
    image.save(f"{save_dir}/{slug}_{name}.png")


def run_model(name, config, prompt, save_dir, negative_prompt=None):
    pretrained_kwargs = {"torch_dtype": torch.float16}
    if "variant" in config:
        pretrained_kwargs["variant"] = config["variant"]
    pipe = config["pipeline"].from_pretrained(
        config["model_id"], **pretrained_kwargs
    ).to("cuda")
    neg = negative_prompt if config.get("negative_prompt") else None
    image = pipe(prompt, negative_prompt=neg, **config["kwargs"]).images[0]
    slug = prompt[:40].replace(" ", "_").replace("/", "-")
    image.save(f"{save_dir}/{slug}_{name}.png")


def main():
    args = parse_args()

    for name in args.models:
        config = MODEL_CONFIGS[name]
        print(f"Running {name} ({config['model_id']})")

        for prompt in args.prompts:
            if name == "sdxl":
                run_sdxl(config, prompt, args.save_dir, args.negative_prompt)
            elif "flux2-dev-4bit" in name:
                run_flux2_4bit(name, config, prompt, args.save_dir, args.negative_prompt)
            else:
                run_model(name, config, prompt, args.save_dir, args.negative_prompt)


if __name__ == "__main__":
    main()
