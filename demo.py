# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dinov3-in1k-probes @ git+https://github.com/yberreby/dinov3-in1k-probes.git",
#     "timm>=1.0",
#     "transformers>=4.50",
# ]
# ///
"""Demo: DINOv3 ImageNet-1k classification with pretrained linear probe."""

print("Importing dependencies...", end=" ", flush=True)
import timm
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from dinov3_in1k_probes import DINOv3LinearClassificationHead

print("done")

VARIANT = "vitb16"
IMAGE_SIZE = 512
PROBE_REPO = f"yberreby/dinov3-{VARIANT}-lvd1689m-in1k-{IMAGE_SIZE}x{IMAGE_SIZE}-linear-clf-probe"
DINOV3_SLUG = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TOP_K = 5

print(f"Loading linear probe: {VARIANT} @ {IMAGE_SIZE}x{IMAGE_SIZE}...", end=" ", flush=True)
probe = DINOv3LinearClassificationHead.from_pretrained(PROBE_REPO)
print("done")

print(f"Loading DINOv3 model: {DINOV3_SLUG}...", end=" ", flush=True)
processor = AutoImageProcessor.from_pretrained(DINOV3_SLUG)
model = AutoModel.from_pretrained(DINOV3_SLUG)
print("done")
print(f"  Patch size: {model.config.patch_size}")
print(f"  Register tokens: {model.config.num_register_tokens}")

print(f"Processing image: {IMAGE_URL}...", end=" ", flush=True)
image = load_image(IMAGE_URL)
inputs = processor(images=image, return_tensors="pt")
print("done")
print(f"  Image size: {image.width}x{image.height}")
print(f"  Preprocessed: {tuple(inputs.pixel_values.shape)}")

print("Running inference...", end=" ", flush=True)
with torch.inference_mode():
    cls = model(**inputs).last_hidden_state[:, 0, :]
    logits = probe(cls.cpu())
    probs = torch.softmax(logits, dim=-1)
print("done")

ini = timm.data.ImageNetInfo()
topk_idx = logits.topk(TOP_K).indices[0]
topk_probs = probs[0, topk_idx]

print(f"\nTop-{TOP_K} predictions:")
for i, (idx, prob) in enumerate(zip(topk_idx, topk_probs), 1):
    label = ini.index_to_description(idx.item())
    print(f"  {i}. {label:40s} {prob * 100:5.2f}%")
