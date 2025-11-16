# DINOv3 ImageNet-1k Linear Classification Probes

Upon its release in August 2025, [DINOv3](https://github.com/facebookresearch/dinov3) marked a milestone in self-supervised representation learning for image processing.
The 7-billion-parameter flagship model was distilled into a family of smaller ViT and ConvNeXT checkpoints, whose sizes make them much more suitable for most CV tasks.

Sadly, only one ImageNet-1k (IN1k) linear classification probe was released: the one for the 7B model.

Here, we provide IN1k probes for the following backbones:
- `facebook/dinov3-vitb16-pretrain-lvd1689m`:
  - [Official backbone](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
  - [512x512 IN1k probe](https://huggingface.co/yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe)

More probes will be released over time.

## Usage

We recommend using [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yberreby/dinov3-imagenet-1k-probes.git
cd dinov3-imagenet-1k-probes
uv run demo.py
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
