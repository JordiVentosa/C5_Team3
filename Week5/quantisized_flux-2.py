#Started code from: https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md

import torch
import os  # <--- Importante para manejar directorios
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" 
device = "cuda:0"
torch_dtype = torch.bfloat16
save_folder = "./outputs/flux2_marc_prompt/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
)
dit = AutoModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
)
pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()


static_prompt_adri = "IMAGE STYLE: Close taken smartphone photo with potential photographic problems: Possible severe motion blur or out of focus, subject might be partially cut off, the photo can be accidental tilt, it can appear a shaky camera, the lighting might be imprect of have hutter lag blur or poorly framed."
static_prompt_marc = "Positive Style: photo taken by a blind person with a smartphone, slightly blurry, poorly framed, everyday object, indoor, amateur photography, low quality camera. Negative Style: professional photography, studio lighting, perfect composition, sharp focus, high resolution, stock photo"


captions = [
    "A wooden table with a wood grain top featuring a pink phone, a baby bottle, a plate, napkins, and some clothes.",
    "An open book or sheet of paper containing information about a card, featuring multiple sections with purple text boxes, colored columns, and bolded black and white lettering.",
    "A musical CD of Death Cab for Cutie’s Codes and Keys in a black and clear case, sitting on top of a table or desk.",
    "A piece of teal or blue fabric with no lettering or design sitting on a surface.",
    "A person holding a large pink bottle of Suave brand strawberry-scented shampoo or medicine in their hand.",
    "An almost empty clear plastic bottle of Germ-X antibacterial hand sanitizer gel on a white computer desk or counter, with the name 'Malcolm' written on the back in black marker.",
    "A black alarm clock or radio with its antenna extended, located in a room with white walls and a pair of white cabinet doors.",
    "A mid-afternoon, overcast outdoor scene at the corner of Ann Street, showing a paved road with sidewalks, a stop sign, street signs, a mailbox, a wooden lamp post, power lines, fencing, and surrounding trees.",
    "A transparent plastic container of cookies featuring both English and Hebrew text and a nutritional label, sitting on a white countertop.",
    "A close-up of a computer monitor showing a web browser with a Google search and a message from Swagbucks saying 'Congratulations! You've won 11 Swagbucks,' with a prompt to insert the capital letters 'WFI' into a provided box.",
    "A black LG computer monitor or television that is turned off, sitting on a table next to a silver Logitech device with a large knob, speakers, and a USB cord.",
    "A 650g peach and white container of Astro Smooth 'n Fruity yogurt sitting on top of a table.",
    "A red can of Chef Boyardee mini ravioli with beef sitting on a kitchen counter next to a box of Raisin Bran and other food or appliances.",
    "The back of a person wearing a bright, light blue branded t-shirt, showing their arm and elbow.",
    "A bottle of Ensure Plus nutrition milk with a straw coming out of it, sitting on a food tray on a table in a hospital or doctor's office with a patient in a hospital bed in the background."
]


for i, c in enumerate(captions):
    image = pipe(
        prompt=c+" "+static_prompt_marc,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50,
        guidance_scale=5,
    ).images[0]
    
    image.save(os.path.join(save_folder, f"{i:03d}_output_flux2_gs5.png"))