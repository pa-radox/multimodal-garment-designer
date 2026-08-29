import os
import json
import torch
import numpy as np

from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

from src.utils.posemap import kpoint_to_heatmap, get_coco_body25_mapping


# ============================================================
# SETTINGS
# ============================================================

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"
MGD_WEIGHTS = r"C:\mgd_models\dresscode.pth"

DATASET = r"C:\Users\osun2\IdeaProjects\multimodal-garment-designer\assets\data\vitonhd\test"

IMAGE_NAME = "03191_00.jpg"

HEIGHT = 512
WIDTH = 384

DEVICE = "cpu"


# ============================================================
# LOAD YOUR MGD PIPELINE
# ============================================================

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from src.mgd_pipelines.mgd_pipe import MGDPipe

import src.mgd_pipelines.mgd_pipe

config = UNet2DConditionModel.load_config(
    SD_PATH + "/unet"
)

config["in_channels"] = 28

unet = UNet2DConditionModel.from_config(config)

unet.load_state_dict(
    torch.load(
        MGD_WEIGHTS,
        map_location="cpu"
    )
)


text_encoder = CLIPTextModel.from_pretrained(
    SD_PATH,
    subfolder="text_encoder",
    local_files_only=True
)

vae_config = AutoencoderKL.load_config(
    SD_PATH + "/vae"
)

vae = AutoencoderKL.from_config(vae_config)

vae.load_state_dict(
    torch.load(
        SD_PATH + "/vae/diffusion_pytorch_model.bin",
        map_location="cpu"
    )
)

tokenizer = CLIPTokenizer.from_pretrained(
    SD_PATH,
    subfolder="tokenizer",
    local_files_only=True
)

scheduler = DDIMScheduler.from_pretrained(
    SD_PATH,
    subfolder="scheduler",
    local_files_only=True
)

scheduler.set_timesteps(50)


mgd_pipe = MGDPipe(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
)


# ============================================================
# USE THE REPOSITORY'S OWN DATASET PREPROCESSING
# ============================================================

from src.datasets.vitonhd import VitonHDDataset

print("Loading VITON-HD dataset...")

dataset = VitonHDDataset(
    dataroot_path=r"C:\Users\osun2\IdeaProjects\multimodal-garment-designer\assets\data\vitonhd",
    phase="test",
    tokenizer=tokenizer,
    order="paired",
    outputlist=(
        "im_name",
        "image",
        "shape",
        "pose_map",
        "im_mask",
        "inpaint_mask",
        "parse_mask_total",
        "im_sketch",
    ),
    size=(512, 384),
)

print("Dataset loaded!")

sample = dataset[0]

# ============================================================
# PREPARE INPUTS
# ============================================================

image = sample["image"].unsqueeze(0)

pose_map = sample["pose_map"].unsqueeze(0)

sketch = sample["im_sketch"].unsqueeze(0)

mask_image = sample["inpaint_mask"].unsqueeze(0)

# Save original image before MGD changes anything
original_image = to_pil_image((image[0] + 1) / 2)
original_image.save("original.png")
# ============================================================
# TEXT PROMPT
# ============================================================

prompt = "a black and red tank top with decorative patterns"

print()
print("PROMPT:")
print(prompt)

# ============================================================
# GENERATE
# ============================================================

print()
print("Starting MGD generation...")
print("This will be VERY slow on CPU.")

from torchvision.transforms.functional import to_pil_image

mask_debug = mask_image.squeeze(0).squeeze(0)

to_pil_image((mask_debug * 255).byte()).save("debug_mask.png")

result = mgd_pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    pose_map=pose_map,
    sketch=sketch,
    height=512,
    width=384,
    num_inference_steps=30,
    guidance_scale=7.5,
    sketch_cond_rate=1.0,
    start_cond_rate=0,
    output_type="pil",
)

generated = result.images[0]

generated_np = np.array(generated).astype(np.float32)
original_np = np.array(original_image).astype(np.float32)

# Get mask
mask_np = mask_image[0].cpu().numpy()

# Remove extra dimensions
mask_np = np.squeeze(mask_np)

# Make mask 3-channel: (512, 384) -> (512, 384, 3)
mask_np = np.stack([mask_np] * 3, axis=2)

# White mask = generated clothing
# Black mask = original image
final_np = generated_np * mask_np + original_np * (1 - mask_np)

final_np = np.clip(final_np, 0, 255).astype(np.uint8)

Image.fromarray(final_np).save(prompt + ".png")