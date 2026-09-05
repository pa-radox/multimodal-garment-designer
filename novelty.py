import os
import re
import torch
import numpy as np

from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel

from src.mgd_pipelines.mgd_pipe import MGDPipe


# ============================================================
# SETTINGS
# ============================================================

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"

# IMPORTANT:
# This should ideally be the VITON-HD MGD checkpoint when
# running on the VITON-HD dataset.
MGD_WEIGHTS = r"C:\mgd_models\dresscode.pth"

DATA_ROOT = (
    r"C:\Users\osun2\IdeaProjects"
    r"\multimodal-garment-designer"
    r"\assets\data\vitonhd"
)

OUTPUT_DIR = "mgd_experiment2"

PERSON_INDEX = 0

NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5

# Keep this fixed so results are reproducible.
BASE_SEED = 1234


# ============================================================
# EXPERIMENT PROMPTS
# ============================================================
#
# The SAME person and SAME garment sketch are used for every
# generation.
#
# Only the textual design description changes.
#
# This allows us to test how much design variation MGD can
# produce from the same initial garment concept.
#

PROMPTS = [
    "a frilly black blouse with transparent black long sleeves, gothic style",

    "a frilly white blouse with transparent white long sleeves, romantic style",

    "a fitted red blouse with puff sleeves and floral patterns",

    "a loose cream blouse with embroidered flowers and ruffled sleeves",

    "a dark blue blouse with geometric patterns and wide sleeves",

    "a pastel pink blouse with lace details and short puff sleeves",

    "a green blouse with asymmetrical details and layered fabric",

    "a black blouse with silver decorative patterns and bell sleeves",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MGD UNET
# ============================================================

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

unet.eval()

print("MGD UNet loaded!")


# ============================================================
# LOAD STABLE DIFFUSION COMPONENTS
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

scheduler.set_timesteps(NUM_INFERENCE_STEPS)


# ============================================================
# CREATE PIPELINE
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
# LOAD VITON-HD DATASET
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
        "im_sketch",
    ),

    size=(512, 384),
)

print("Dataset loaded!")


# ============================================================
# GET PERSON / SKETCH
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


image = sample["image"].unsqueeze(0)

pose_map = sample["pose_map"].unsqueeze(0)

mask_image = sample["inpaint_mask"].unsqueeze(0)

# IMPORTANT:
#
# Because order="unpaired", the dataset internally loads
# the sketch from im_sketch_unpaired.
#
# It is returned to us under the key "im_sketch".

sketch = sample["im_sketch"].unsqueeze(0)


# ============================================================
# SAVE ORIGINAL INPUTS
# ============================================================

original_np = (
    image[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

original_np = ((original_np + 1) / 2 * 255)

original_np = np.clip(
    original_np,
    0,
    255
).astype(np.uint8)

Image.fromarray(
    original_np
).save(
    os.path.join(
        OUTPUT_DIR,
        "original_person.png"
    )
)


# ============================================================
# SAVE MASK
# ============================================================

mask_debug = mask_image[0].cpu()

if mask_debug.ndim == 3:
    mask_debug = mask_debug.squeeze(0)

mask_debug = (
        mask_debug.numpy() * 255
)

mask_debug = np.clip(
    mask_debug,
    0,
    255
).astype(np.uint8)

Image.fromarray(
    mask_debug
).save(
    os.path.join(
        OUTPUT_DIR,
        "mask.png"
    )
)


# ============================================================
# SAVE SKETCH
# ============================================================

sketch_debug = sketch[0].cpu()

if sketch_debug.shape[0] == 1:
    sketch_debug = sketch_debug.squeeze(0)

sketch_debug = (
        sketch_debug.numpy() * 255
)

sketch_debug = np.clip(
    sketch_debug,
    0,
    255
).astype(np.uint8)

Image.fromarray(
    sketch_debug
).save(
    os.path.join(
        OUTPUT_DIR,
        "garment_sketch.png"
    )
)


# ============================================================
# PREPARE ORIGINAL IMAGE FOR COMPOSITING
# ============================================================

original_np = (
    image[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

original_np = (original_np + 1) / 2
original_np = original_np * 255

original_np = np.clip(
    original_np,
    0,
    255
)


# ============================================================
# PREPARE MASK FOR COMPOSITING
# ============================================================

mask_np = mask_image[0].cpu().numpy()

if mask_np.ndim == 3:
    mask_np = mask_np.squeeze(0)

mask_np = mask_np[:, :, None]


# ============================================================
# GENERATION LOOP
# ============================================================

print()
print("========================================")
print("STARTING MGD DESIGN DIVERSITY EXPERIMENT")
print("========================================")
print()

print(
    "Number of designs:",
    len(PROMPTS)
)

print(
    "Inference steps:",
    NUM_INFERENCE_STEPS
)

print(
    "Base seed:",
    BASE_SEED
)

print()
print("This will be VERY slow on CPU.")
print()


for i, prompt in enumerate(PROMPTS):

    print()
    print("----------------------------------------")
    print(
        f"GENERATING DESIGN {i + 1}/{len(PROMPTS)}"
    )
    print("----------------------------------------")

    print("Prompt:")
    print(prompt)

    # --------------------------------------------------------
    # REPRODUCIBLE SEED
    # --------------------------------------------------------

    seed = BASE_SEED + i

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Seed:", seed)


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

        sketch_cond_rate=0.5,
        start_cond_rate=0,

        output_type="pil",
    )


    generated = result.images[0]


    # --------------------------------------------------------
    # SAVE RAW MGD OUTPUT
    # --------------------------------------------------------

    raw_filename = (
        f"design_{i + 1:02d}_raw.png"
    )

    generated.save(
        os.path.join(
            OUTPUT_DIR,
            raw_filename
        )
    )


    # --------------------------------------------------------
    # COMPOSITE WITH ORIGINAL PERSON
    #
    # Generated clothing stays inside the mask.
    # Everything outside the mask comes from the original
    # person image.
    # --------------------------------------------------------

    generated_np = np.array(
        generated
    ).astype(np.float32)


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


    final_image = Image.fromarray(
        final_np
    )


    # --------------------------------------------------------
    # SAFE FILENAME
    # --------------------------------------------------------

    safe_prompt = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        prompt
    ).strip("_")


    filename = (
        f"design_{i + 1:02d}_"
        f"{safe_prompt}.png"
    )


    final_image.save(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )


    print(
        "Saved:",
        filename
    )


print()
print("========================================")
print("EXPERIMENT COMPLETE")
print("========================================")
print()
print(
    "Results saved to:",
    OUTPUT_DIR
)
