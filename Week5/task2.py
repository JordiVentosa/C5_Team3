import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42

# Blind photography style descriptors
BLIND_SUFFIX = (
    ""
    # "blurry, out of focus, off-center subject, tilted angle, accidental framing, "
    # "amateur photo, shaky camera, partially visible subject, bad composition"
)
BLIND_NEGATIVE = "sharp focus, centered, professional photography, well-composed, high quality, 8k"


def load_pipeline(device="cuda"):
    dtype = torch.float32 if device == "cpu" else torch.float16
    kwargs = {} if device == "cpu" else {"variant": "fp16"}
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        **kwargs,
    ).to(device)
    return pipe


def reset_scheduler(pipe):
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


def save(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(f"Saved: {path}")


DEVICE = "cuda"

def generator(seed=SEED):
    return torch.Generator(DEVICE).manual_seed(seed)


def blind_prompt(prompt):
    return f"{prompt}, {BLIND_SUFFIX}"


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ─── Experiment 1: Sampler (DDPM vs DDIM vs default FlowMatch) ────────────────
# Goal: DDPM stochastic noise → more accidental/random look at low steps
# DDIM deterministic → cleaner degradation curve
# FlowMatch: SD3.5 default, straightest trajectories

def exp_sampler(pipe, prompt, save_dir):
    out = os.path.join(save_dir, "exp1_sampler")
    prompt = blind_prompt(prompt)

    sdxl_kwargs = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    schedulers = {
        "euler": None,  # SDXL default
        "ddim": DDIMScheduler(**sdxl_kwargs),
        "ddpm": DDPMScheduler(**sdxl_kwargs),
    }

    # Quality-vs-steps comparison: DDIM can match DDPM quality in far fewer steps
    comparisons = [
        ("ddpm", schedulers["ddpm"], 1000),  # correct DDPM usage — slow but reference quality
        ("ddpm", schedulers["ddpm"],   20),  # DDPM shortcut — visibly worse
        ("ddim", schedulers["ddim"],   20),  # DDIM matches DDPM@1000 in 20 steps
        ("ddim", schedulers["ddim"],   10),
        ("euler", None,                20),
    ]
    for name, sched, steps in comparisons:
        pipe.scheduler = sched if sched is not None else EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        print(f"  {type(pipe.scheduler).__name__} @ {steps} steps")
        image = pipe(
            prompt,
            negative_prompt=BLIND_NEGATIVE,
            num_inference_steps=steps,
            guidance_scale=4.5,
            generator=generator(),
        ).images[0]
        save(image, f"{out}/{name}_steps{steps:03d}.png")


# ─── Experiment 2: Number of denoising steps ─────────────────────────────────
# Goal: few steps = naturally blurry/unfinished → mimics blind photo aesthetics
# Shows how incomplete denoising produces the desired visual degradation

def exp_steps(pipe, prompt, save_dir):
    out = os.path.join(save_dir, "exp2_steps")
    reset_scheduler(pipe)
    prompt = blind_prompt(prompt)

    for steps in [5, 10, 20, 30, 40, 60]:
        image = pipe(
            prompt,
            negative_prompt=BLIND_NEGATIVE,
            num_inference_steps=steps,
            guidance_scale=4.5,
            generator=generator(),
        ).images[0]
        save(image, f"{out}/steps{steps}.png")


# ─── Experiment 3: CFG strength ───────────────────────────────────────────────
# Goal: low CFG = model ignores prompt → unpredictable, accidental framing
# High CFG = over-sharpened, too composed — opposite of blind photo style
# Sweet spot: ~2-3 for loose, accidental look

def exp_cfg(pipe, prompt, save_dir):
    out = os.path.join(save_dir, "exp3_cfg")
    reset_scheduler(pipe)
    prompt = blind_prompt(prompt)

    for cfg in [1.0, 2.0, 3.0, 4.5, 7.0, 10.0]:
        image = pipe(
            prompt,
            negative_prompt=BLIND_NEGATIVE,
            num_inference_steps=40,
            guidance_scale=cfg,
            generator=generator(),
        ).images[0]
        save(image, f"{out}/cfg{cfg}.png")


# ─── Experiment 4: Positive & negative prompting ─────────────────────────────
# Goal: explore how prompting intensity controls blind photography aesthetics
# Rows: how much blind-style description we add to positive prompt
# Cols: how strongly we push away sharp/professional look via negative prompt

def exp_prompting(pipe, prompt, save_dir):
    out = os.path.join(save_dir, "exp4_prompting")
    reset_scheduler(pipe)

    positive_variants = {
        "bare":          prompt,
        "stylized":      f"{prompt}, cinematic lighting, oil painting, dramatic atmosphere",
        "degraded":      f"{prompt}, blurry, out of focus, off-center, tilted angle, shaky camera",
        "detailed":      f"{prompt}, sharp focus, 8k, professional DSLR, golden hour, photorealistic",
        "contradictory": f"{prompt}, blurry, off-center, shaky camera, 8k, professional DSLR, sharp",
    }

    negative_variants = {
        "none":      None,
        "quality":   "blurry, low resolution, noisy, grainy, pixelated, jpeg artifacts",
        "style":     "painting, cartoon, illustration, drawing, anime, sketch, CGI",
        "scene":     "indoor, ground level, floor, interior, daytime",
        "combined":  "blurry, low resolution, noisy, grainy, painting, cartoon, illustration, indoor, ground level",
    }

    for pos_name, pos_prompt in positive_variants.items():
        for neg_name, neg_prompt in negative_variants.items():
            image = pipe(
                pos_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=40,
                guidance_scale=7.0,
                generator=generator(),
            ).images[0]
            save(image, f"{out}/{pos_name}__{neg_name}.png")


# ─── Experiment 5: Denoising step visualization ───────────────────────────────
# Goal: save decoded image at every denoising step
# Shows how noise → structure → final image across the diffusion process

def exp_denoising_viz(pipe, prompt, save_dir):
    out = os.path.join(save_dir, "exp5_denoising_viz")
    os.makedirs(out, exist_ok=True)
    reset_scheduler(pipe)
    prompt = blind_prompt(prompt)

    def step_callback(pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        image = decode_latents(pipeline, latents.clone())
        image.save(f"{out}/step_{step:03d}_t{int(timestep):04d}.png")
        print(f"  step {step}, t={int(timestep)}")
        return callback_kwargs

    pipe(
        prompt,
        negative_prompt=BLIND_NEGATIVE,
        num_inference_steps=40,
        guidance_scale=4.5,
        generator=generator(),
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "sampler": exp_sampler,
    "steps": exp_steps,
    "cfg": exp_cfg,
    "prompting": exp_prompting,
    "denoising_viz": exp_denoising_viz,
}


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL blind photography experiments")
    parser.add_argument("--prompt", type=str, required=True, help="Base scene description.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help=f"Experiments to run. Available: {', '.join(EXPERIMENTS.keys())}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global DEVICE
    DEVICE = args.device
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    pipe = load_pipeline(args.device)

    for exp_name in args.experiments:
        print(f"\n=== Experiment: {exp_name} ===")
        EXPERIMENTS[exp_name](pipe, args.prompt, args.save_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
