# DINOv3 ImageNet-1k Linear Classification Probes

Upon its release in August 2025, [DINOv3](https://github.com/facebookresearch/dinov3) marked a milestone in self-supervised representation learning for image processing.
The 7-billion-parameter flagship model was distilled into a family of smaller ViT and ConvNeXT checkpoints, whose sizes make them much more suitable for most CV tasks.

Sadly, only one ImageNet-1k (IN1k) linear classification probe was released: the one for the 7B model.

**Here, we release pretrained linear probes for some of the smaller DINOv3 ViT models.**
They can be used directly with Meta's official checkpoints.

As in the original DINOv3 paper, we used **512x512 inputs** (1024 input tokens),
and trained the probes on the IN1k training set with Inception-crop augmentation.

Thanks to a highly-optimized training pipeline, we were able to perform more extensive hyperparameter sweeps than in the original paper.
As a result, **all of our probes match or exceed the best IN1k-ReAL top-1 validation accuracy reported by the DINOv3 authors**, as seen in Table 14 of the original paper.

We note that the raw IN1k top-1 validation accuracy was not reported by the DINOv3 authors, only the [ReAL](https://github.com/google-research/reassessed-imagenet) top-1 accuracy.
Here, we report both.


## Released Probes

- **ViT-S/16** @ 512×512
  - Base: [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
  - Probe: [`yberreby/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/yberreby/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-B/16** @ 512×512
  - Base: [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
  - Probe: [`yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-L/16** @ 512×512
  - Base: [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  - Probe: [`yberreby/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/yberreby/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe)

More probes will be released over time; watch this repository and/or [the corresponding HuggingFace Collection](https://huggingface.co/collections/yberreby/dinov3-imagenet-1k-probes).

## Performance

Thanks to a high-performance training pipeline, we were able to sweep 

| Probe | [IN-ReAL](https://github.com/google-research/reassessed-imagenet) top-1 (official / ours) | IN1k top-1 (ours) |
|-------|--------------------------------|-------------------|
| [ViT-S/16](https://huggingface.co/yberreby/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe) | 87.0% / **87.09%** | 81.29% |
| [ViT-B/16](https://huggingface.co/yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe) | 89.3% / **89.54%** | 85.00% |
| [ViT-L/16](https://huggingface.co/yberreby/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe) | 90.2% / **90.44%** | 86.70% |
| ViT-S+/16 (coming soon) | 88.0% / — |  -- |
| ViT-H+/16 (coming soon) | 90.3% / — |  -- |

The accuracy of the latest probes uploaded on the HF Hub can be queried using `uv run print_metrics.py`.

## Usage

We recommend using [`uv`](https://docs.astral.sh/uv/).

### Using `from_pretrained`

```
❯ uvx --with 'https://github.com/yberreby/dinov3-imagenet-1k-probes.git' ipython
    Updated https://github.com/yberreby/dinov3-imagenet-1k-probes.git (41a0d0b86dbbe3b8f7b3a776fe0aa5f1fb46f635)
      Built dinov3-probes @ git+https://github.com/yberreby/dinov3-imagenet-1k-probes.git@41a0d0b86dbbe3b8f7b3a776fe0aa5f1fb46f635
Installed 44 packages in 225ms
Python 3.12.11 (main, Jul  8 2025, 20:41:49) [Clang 20.1.4 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 9.7.0 -- An enhanced Interactive Python. Type '?' for help.
Tip: The `%timeit` magic has a `-o` flag, which returns the results, making it easy to plot. See `%timeit?`.

In [1]: from dinov3_probes import DINOv3LinearClassificationHead

In [2]: probe = DINOv3LinearClassificationHead.from_pretrained("yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe")

In [3]: probe
Out[3]: DINOv3LinearClassificationHead(in_features=768, out_features=1000, bias=True)
```

### Detailed example

```bash
git clone https://github.com/yberreby/dinov3-imagenet-1k-probes.git
cd dinov3-imagenet-1k-probes
uv run demo.py # or `uv run ipython -i demo.py` for a REPL
```

Example:
```
dinov3-imagenet-1k-probes on  main [?] is 📦 v0.1.0 via 🐍 v3.13.8
❯ uv run ipython -i demo.py
Python 3.12.11 (main, Jul  8 2025, 20:41:49) [Clang 20.1.4 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 9.7.0 -- An enhanced Interactive Python. Type '?' for help.
Tip: You can use Ctrl-O to force a new line in terminal IPython
Importing dependencies... done
Loading linear probe: vitb16 @ 512x512... done
  IN1k val top-1: 85.00%
  IN1k-ReAL top-1: 89.54%

Loading DINOv3 model: facebook/dinov3-vitb16-pretrain-lvd1689m... done
  Patch size: 16
  Register tokens: 4

Processing image: http://images.cocodataset.org/val2017/000000039769.jpg... done
  Image size: 640x480
  Preprocessed: (1, 3, 224, 224)

Running inference... done


Top-5 predictions:
  1. tabby, tabby cat                         48.01%
  2. Egyptian cat                             18.62%
  3. tiger cat                                10.51%
  4. remote control, remote                    6.67%
  5. mouse, computer mouse                     1.58%

In [1]: probe
Out[1]: Linear(in_features=768, out_features=1000, bias=True)

In [2]: model
Out[2]:
DINOv3ViTModel(
  (embeddings): DINOv3ViTEmbeddings(
    (patch_embeddings): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  )
  (rope_embeddings): DINOv3ViTRopePositionEmbedding()
  (layer): ModuleList(
    (0-11): 12 x DINOv3ViTLayer(
      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attention): DINOv3ViTAttention(
        (k_proj): Linear(in_features=768, out_features=768, bias=False)
        (v_proj): Linear(in_features=768, out_features=768, bias=True)
        (q_proj): Linear(in_features=768, out_features=768, bias=True)
        (o_proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (layer_scale1): DINOv3ViTLayerScale()
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): DINOv3ViTMLP(
        (up_proj): Linear(in_features=768, out_features=3072, bias=True)
        (down_proj): Linear(in_features=3072, out_features=768, bias=True)
        (act_fn): GELUActivation()
      )
      (layer_scale2): DINOv3ViTLayerScale()
    )
  )
  (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)

In [3]: probs.shape
Out[3]: torch.Size([1, 1000])

In [4]:
```


## Development

To push to HuggingFace Hub, use the `push_to_hub.py` script.
It will auto-detect the model name, image size, and other metadata.

Example usage:

```bash
uv run push_to_hub.py --checkpoint dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe.pt
```
