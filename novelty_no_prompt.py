import os
import torch
import numpy as np

from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel

from src.mgd_pipelines.mgd_pipe import MGDPipe


# ------------------------------------------------------------
# EXPERIMENT SETTINGS
# ------------------------------------------------------------

NUM_VARIATIONS = 3

NUM_INFERENCE_STEPS = 20

GUIDANCE_SCALE = 7.5

# Same base seed every time you run the experiment.
# Each variation gets BASE_SEED + variation number.
BASE_SEED = 1003

# ------------------------------------------------------------
# SKETCH CONDITIONING
# ------------------------------------------------------------
#
# 1.0 = strongest sketch conditioning
# 0.5 = weaker sketch conditioning
# 0.25 = much weaker sketch conditioning
#
# Start with 1.0 for the actual experiment.
#

SKETCH_COND_RATE = 0.75
START_COND_RATE = 0.25

# ============================================================
# SETTINGS
# ============================================================

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"

MGD_WEIGHTS = r"C:\mgd_models\dresscode.pth"

DATA_ROOT = (
    r"C:\Users\osun2\IdeaProjects"
    r"\multimodal-garment-designer"
    r"\assets\data\vitonhd"
)

#OUTPUT_DIR = "mgd_variations_" + str(SKETCH_COND_RATE) + "_" + str(START_COND_RATE)
OUTPUT_DIR = "specific_seed_" + str(SKETCH_COND_RATE) + "_" + str(START_COND_RATE)

PERSON_INDEX = 0

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MGD UNET
# ============================================================

print("Loading MGD UNet...")

config = UNet2DConditionModel.load_config(
    SD_PATH + "/unet"
)

config["in_channels"] = 28

unet = UNet2DConditionModel.from_config(
    config
)

unet.load_state_dict(
    torch.load(
        MGD_WEIGHTS,
        map_location="cpu"
    )
)

unet.eval()

print("MGD UNet loaded!")


# ============================================================
# LOAD STABLE DIFFUSION
# ============================================================

print("Loading Stable Diffusion components...")

text_encoder = CLIPTextModel.from_pretrained(
    SD_PATH,
    subfolder="text_encoder",
    local_files_only=True
)

vae_config = AutoencoderKL.load_config(
    SD_PATH + "/vae"
)

vae = AutoencoderKL.from_config(
    vae_config
)

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

scheduler.set_timesteps(
    NUM_INFERENCE_STEPS
)


# ============================================================
# CREATE MGD PIPELINE
# ============================================================

print("Creating MGD pipeline...")

mgd_pipe = MGDPipe(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
)

print("MGD PIPELINE LOADED!")


# ============================================================
# LOAD VITON-HD
# ============================================================

print("Loading VITON-HD dataset...")

from src.datasets.vitonhd import VitonHDDataset

dataset = VitonHDDataset(
    dataroot_path=DATA_ROOT,
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

# ============================================================
# SELECT PERSON
# ============================================================

sample = dataset[PERSON_INDEX]

print(
    "Using person sample:",
    PERSON_INDEX
)

print(
    "Image:",
    sample["im_name"]
)

# ============================================================
# INPUTS
# ============================================================

image = sample["image"].unsqueeze(0)
pose_map = sample["pose_map"].unsqueeze(0)
mask_image = sample["inpaint_mask"].unsqueeze(0)

# Because order="unpaired", this is the
# im_sketch_unpaired image internally.
sketch = sample["im_sketch"].unsqueeze(0)

# ============================================================
# SAVE INPUTS
# ============================================================

original_np = (image[0].permute(1, 2, 0).cpu().numpy())
original_np = (original_np + 1) / 2 * 255
original_np = np.clip(original_np,0,255).astype(np.uint8)

Image.fromarray(original_np).save(
    os.path.join(
        OUTPUT_DIR,
        "original_person.png"
    )
)

# ------------------------------------------------------------
# MASK
# ------------------------------------------------------------

mask_debug = mask_image[0].cpu()

if mask_debug.ndim == 3:
    mask_debug = mask_debug.squeeze(0)

mask_debug = mask_debug.numpy() * 255

mask_debug = np.clip(mask_debug,0,255).astype(np.uint8)

Image.fromarray(mask_debug).save(
    os.path.join(
        OUTPUT_DIR,
        "mask.png"
    )
)

# ------------------------------------------------------------
# SKETCH
# ------------------------------------------------------------

sketch_debug = sketch[0].cpu()

if sketch_debug.shape[0] == 1:
    sketch_debug = sketch_debug.squeeze(0)

sketch_debug = sketch_debug.numpy() * 255

sketch_debug = np.clip(sketch_debug,0,255).astype(np.uint8)

Image.fromarray(sketch_debug).save(
    os.path.join(
        OUTPUT_DIR,
        "garment_sketch.png"
    )
)

# ============================================================
# ORIGINAL IMAGE FOR COMPOSITING
# ============================================================

original_np = image[0].permute(1, 2, 0).cpu().numpy()
original_np = (original_np + 1) / 2 * 255
original_np = np.clip(original_np,0,255)

# ============================================================
# MASK FOR COMPOSITING
# ============================================================

mask_np = mask_image[0].cpu().numpy()

if mask_np.ndim == 3:
    mask_np = mask_np.squeeze(0)

mask_np = mask_np[:, :, None]

# ============================================================
# TEXT PROMPT
# ============================================================

prompt = "a fashionable garment with distinctive and unique design details"

# ============================================================
# START EXPERIMENT
# ============================================================

print()
print("========================================")
print("MGD DESIGN DIVERSITY EXPERIMENT")
print("========================================")
print()

print("Number of variations:",NUM_VARIATIONS)

print("Prompt:",prompt)

print("Sketch conditioning:",SKETCH_COND_RATE)

print("Inference steps:",NUM_INFERENCE_STEPS)

print("Base seed:",BASE_SEED)

print()
print("This will be VERY slow on CPU.")
print()

# ============================================================
# GENERATE VARIATIONS
# ============================================================

for i in range(NUM_VARIATIONS):
    variation_number = i + 1
    seed = BASE_SEED + i

    print()
    print("----------------------------------------")
    print(
        f"VARIATION {variation_number}/{NUM_VARIATIONS}"
    )
    print("----------------------------------------")

    print("Seed:",seed)

    # --------------------------------------------------------
    # SET RANDOM SEED
    # --------------------------------------------------------

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    result = mgd_pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        pose_map=pose_map,
        sketch=sketch,
        height=512,
        width=384,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        sketch_cond_rate=SKETCH_COND_RATE,
        start_cond_rate=START_COND_RATE,
        output_type="pil",
    )

    generated = result.images[0]

    # --------------------------------------------------------
    # SAVE RAW OUTPUT
    # --------------------------------------------------------

    raw_filename = (
        f"variation_{variation_number:02d}_raw.png"
    )

    generated.save(
        os.path.join(
            OUTPUT_DIR,
            raw_filename
        )
    )

    # --------------------------------------------------------
    # COMPOSITE
    # --------------------------------------------------------

    generated_np = np.array(generated).astype(np.float32)
    final_np = generated_np * mask_np+original_np * (1 - mask_np)
    final_np = np.clip(final_np,0,255).astype(np.uint8)

    final_image = Image.fromarray(final_np)

    # --------------------------------------------------------
    # SAVE FINAL IMAGE
    # --------------------------------------------------------

    filename = (f"variation_{variation_number:02d}.png")

    final_image.save(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

    print("Saved:",filename)


# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("EXPERIMENT COMPLETE")
print("========================================")
print()

print("Results saved to:",OUTPUT_DIR)

print()
print("Generated",NUM_VARIATIONS,"variations.")