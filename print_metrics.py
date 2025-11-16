import json

from huggingface_hub import hf_hub_download
from tabulate import tabulate

# ============================================================================
# CONFIG
# ============================================================================
VARIANTS = ["vits16", "vitb16", "vitl16"]
IMAGE_SIZES = [512]

# ============================================================================
# FETCH AND PRINT METRICS
# ============================================================================
rows = []
for variant in VARIANTS:
    for image_size in IMAGE_SIZES:
        repo = f"yberreby/dinov3-{variant}-lvd1689m-in1k-{image_size}x{image_size}-linear-clf-probe"
        cfg_path = hf_hub_download(repo, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        vr = cfg["val_results"]
        rows.append(
            [
                repo,
                f"{vr['top1'] * 100:.2f}%",
                f"{vr['real_top1'] * 100:.2f}%",
            ]
        )

headers = ["HF Hub Repo", "IN1k val top-1", "IN-ReAL top-1"]
print(tabulate(rows, headers=headers, tablefmt="github"))
