import json

print("Importing dependencies...", end=" ", flush=True)
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

print("done")

# ============================================================================
# CONFIG
# ============================================================================
VARIANT = "vitb16"
IMAGE_SIZE = 512
PROBE_REPO = f"yberreby/dinov3-{VARIANT}-lvd1689m-in1k-{IMAGE_SIZE}x{IMAGE_SIZE}-linear-clf-probe"
DINOV3_SLUG = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TOP_K = 5

# ============================================================================
# LOAD PROBE
# ============================================================================
print(
    f"Loading linear probe: {VARIANT} @ {IMAGE_SIZE}x{IMAGE_SIZE}...",
    end=" ",
    flush=True,
)
with open(hf_hub_download(PROBE_REPO, "config.json")) as f:
    cfg = json.load(f)
probe = nn.Linear(cfg["in_features"], cfg["out_features"])
probe.load_state_dict(load_file(hf_hub_download(PROBE_REPO, "model.safetensors")))
print("done")
print(f"  IN1k val top-1: {cfg['val_results']['top1'] * 100:.2f}%")
print(f"  IN1k-ReAL top-1: {cfg['val_results']['real_top1'] * 100:.2f}%")
print()

# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"Loading DINOv3 model: {DINOV3_SLUG}...", end=" ", flush=True)
processor = AutoImageProcessor.from_pretrained(DINOV3_SLUG)
model = AutoModel.from_pretrained(DINOV3_SLUG)
print("done")
print(f"  Patch size: {model.config.patch_size}")
print(f"  Register tokens: {model.config.num_register_tokens}")
print()

# ============================================================================
# INFERENCE
# ============================================================================
print(f"Processing image: {IMAGE_URL}...", end=" ", flush=True)
image = load_image(IMAGE_URL)
inputs = processor(images=image, return_tensors="pt")
print("done")
print(f"  Image size: {image.width}x{image.height}")
print(f"  Preprocessed: {tuple(inputs.pixel_values.shape)}")
print()

print("Running inference...", end=" ", flush=True)
with torch.inference_mode():
    cls = model(**inputs).last_hidden_state[:, 0, :]
    logits = probe(cls.cpu())
    probs = torch.softmax(logits, dim=-1)
print("done")
print()

# ============================================================================
# RESULTS
# ============================================================================
ini = timm.data.ImageNetInfo()
topk_idx = logits.topk(TOP_K).indices[0]
topk_probs = probs[0, topk_idx]

print(f"\nTop-{TOP_K} predictions:")
for i, (idx, prob) in enumerate(zip(topk_idx, topk_probs), 1):
    label = ini.index_to_description(idx.item())
    print(f"  {i}. {label:40s} {prob * 100:5.2f}%")
