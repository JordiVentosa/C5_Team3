import torch
from diffusers import Flux2Pipeline
from diffusers import Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration

MODEL_PATH = "FLUX/FLUX.2-dev"
OUTPUT_PATH = "outputs/frog.png"

PROMPT = "A realistic image of a frog riding a motorcicle on space"

torch_dtype = torch.bfloat16

print("Loading model...")
transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer",
    torch_dtype=torch_dtype, device_map="auto",
)
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_PATH, subfolder="text_encoder",
    torch_dtype=torch_dtype, device_map="auto",
)
pipe = Flux2Pipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch_dtype,
)
pipe.vae = pipe.vae.to("cuda:0")

print("Model loaded. Generating...")

image = pipe(
    prompt=PROMPT,
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator(device="cuda:0").manual_seed(42),
).images[0]

image.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")