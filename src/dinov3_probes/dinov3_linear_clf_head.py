import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class DINOv3LinearClassificationHead(
    nn.Linear,
    PyTorchModelHubMixin,
    library_name="dinov3-probes",
    repo_url="https://github.com/yberreby/dinov3-probes",
    paper_url="see the original DINOv3 paper at https://arxiv.org/abs/2508.10104",
    docs_url="https://github.com/yberreby/dinov3-probes",
):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
