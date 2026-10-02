import sys, types, torch
from src.models.modeling import VLMEncoder

MODELS = sys.argv[1:] or [
    "OpenCLIP-ViT-B-16", "OpenCLIP-ViT-L-14", "OpenCLIP-ViT-H-14", "ConvNeXt-B",
    "SigLIP-ViT-B-16", "SigLIP2-ViT-B-16", "MetaCLIP-ViT-B-16", "DataComp-ViT-B-16",
]
for name in MODELS:
    enc = VLMEncoder(types.SimpleNamespace(model=name, device="cpu"))
    bias = getattr(enc.model, "logit_bias", None)
    print(f"{name:22s} scale={enc.model.logit_scale.exp().item():.4f}  "
          f"bias={None if bias is None else bias.item()}")
    del enc