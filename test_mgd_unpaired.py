import os
import torch
import numpy as np

from torchvision.transforms.functional import to_pil_image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel

from src.mgd_pipelines.mgd_pipe import MGDPipe

# PATHS

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"
MGD_WEIGHTS = r"C:\mgd_models\dresscode.pth"

# LOAD MGD UNET

print("Loading MGD UNet...")

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

print("MGD UNet loaded!")

# LOAD STABLE DIFFUSION COMPONENTS

print("Loading Stable Diffusion components...")

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

# CREATE PIPELINE

print("Creating MGD pipeline...")

mgd_pipe = MGDPipe(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
)

print("MGD PIPELINE LOADED!")

# LOAD DATASET

print("Loading VITON-HD dataset...")

from src.datasets.vitonhd import VitonHDDataset

dataset = VitonHDDataset(
    dataroot_path=r"C:\Users\osun2\IdeaProjects\multimodal-garment-designer\assets\data\vitonhd",
    phase="test",
    tokenizer=tokenizer,
    order="unpaired",
    outputlist=(
        "im_name",
        "image",
        "shape",
        "pose_map",
        "im_mask",
        "inpaint_mask",
        "parse_mask_total",
        "im_sketch"
    ),
    size=(512, 384),
)

print("Dataset loaded!")

# CHOOSE SAMPLE

person_index = 0

sample = dataset[person_index]

print("Using person sample:", person_index)

# INPUTS FROM PERSON

image = sample["image"].unsqueeze(0)

pose_map = sample["pose_map"].unsqueeze(0)

mask_image = sample["inpaint_mask"].unsqueeze(0)

# UNPAIRED SKETCH

# MGD's VITON-HD data contains a separate unpaired sketch.
#
# If your dataset class exposes it as "im_sketch_unpaired",
# this is all we need:

sketch = sample["im_sketch"].unsqueeze(0)

# SAVE DEBUG INPUTS

# Save original person
original_np = image[0].permute(1, 2, 0).numpy()
original_np = ((original_np + 1) / 2 * 255)
original_np = np.clip(original_np, 0, 255).astype(np.uint8)

from PIL import Image

Image.fromarray(original_np).save("unpaired_original.png")

# Save mask
mask_debug = mask_image[0]

if mask_debug.ndim == 3:
    mask_debug = mask_debug.squeeze(0)

mask_debug = (mask_debug.cpu().numpy() * 255)
mask_debug = np.clip(mask_debug, 0, 255).astype(np.uint8)

Image.fromarray(mask_debug).save("unpaired_mask.png")

# Save sketch
sketch_debug = sketch[0]

if sketch_debug.shape[0] == 1:
    sketch_debug = sketch_debug.squeeze(0)

sketch_debug = (sketch_debug.cpu().numpy() * 255)
sketch_debug = np.clip(sketch_debug, 0, 255).astype(np.uint8)

Image.fromarray(sketch_debug).save("unpaired_sketch.png")

# PROMPT

prompt = "a frilly black blouse, transparent black long sleeves goth"

print()
print("Starting UNPAIRED MGD generation...")
print("This will be VERY slow on CPU.")

# ============================================================
# GENERATE
# ============================================================

result = mgd_pipe(
    prompt=prompt,

    image=image,
    mask_image=mask_image,

    pose_map=pose_map,
    sketch=sketch,

    height=512,
    width=384,

    num_inference_steps=20,
    guidance_scale=7.5,

    sketch_cond_rate=1.0,
    start_cond_rate=0,

    output_type="pil",
)

generated = result.images[0]

generated.save("unpaired_mgd_raw.png")

# ============================================================
# COMPOSITE WITH ORIGINAL PERSON
# ============================================================

generated_np = np.array(generated).astype(np.float32)

# Mask currently has shape:
# [1, 1, H, W]
#
# Convert to:
# [H, W, 1]

mask_np = mask_image[0].cpu().numpy()

if mask_np.ndim == 3:
    mask_np = mask_np.squeeze(0)

mask_np = mask_np[:, :, None]

# Convert original image to H,W,3
original_np = image[0].permute(1, 2, 0).cpu().numpy()

# [-1,1] -> [0,255]
original_np = (original_np + 1) / 2
original_np = original_np * 255

original_np = np.clip(
    original_np,
    0,
    255
)

# ============================================================
# COMPOSITE
# ============================================================

final_np = (
        generated_np * mask_np
        +
        original_np * (1 - mask_np)
)

final_np = np.clip(
    final_np,
    0,
    255
).astype(np.uint8)

final_image = Image.fromarray(final_np)

final_image.save(prompt + ".png")

print()
print("FINAL RESULT:")
print(prompt + ".png")
print()
print("Done!")