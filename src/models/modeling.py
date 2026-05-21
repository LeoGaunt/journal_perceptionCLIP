"""
modeling.py — Unified VLM encoder supporting multiple backends.

Supported model strings
-----------------------
Original CLIP (via openai/CLIP):
    ViT-B/16, ViT-L/14@336px, RN50x4, ViT-B/32, RN50, RN101

OpenCLIP — LAION-trained (via mlfoundations/open_clip):
    OpenCLIP-ViT-H-14          (laion2b_s32b_b79k)
    OpenCLIP-ViT-L-14          (laion2b_s32b_b82k)
    OpenCLIP-ViT-B-16          (laion2b_s34b_b88k)

MetaCLIP (via open_clip with metaclip weights):
    MetaCLIP-ViT-B-16          (metaclip_400m)
    MetaCLIP-ViT-L-14          (metaclip_fullcc)
    MetaCLIP-ViT-H-14          (metaclip_fullcc)

SigLIP (via open_clip with webli weights):
    SigLIP-ViT-B-16            (webli)
    SigLIP-ViT-L-16-384        (webli)

Usage
-----
All models expose the same interface:
    encoder.tokenize(texts)          <- USE THIS, never clip.tokenize directly
    encoder.encode_image(images)
    encoder.encode_text(token_ids)
    encoder.logit_scale
    encoder.val_preprocess
"""

import torch
import torch.nn.functional as F

import clip.clip as clip
import open_clip

from src.models import utils


# ---------------------------------------------------------------------------
# Registry: maps model-string -> (open_clip_arch, pretrained_tag)
# ---------------------------------------------------------------------------
_OPENCLIP_REGISTRY = {
    # LAION-trained
    "OpenCLIP-ViT-H-14":      ("ViT-H-14",            "laion2b_s32b_b79k"),
    "OpenCLIP-ViT-L-14":      ("ViT-L-14",            "laion2b_s32b_b82k"),
    "OpenCLIP-ViT-B-16":      ("ViT-B-16",            "laion2b_s34b_b88k"),
    # MetaCLIP
    "MetaCLIP-ViT-B-16":      ("ViT-B-16",            "metaclip_400m"),
    "MetaCLIP-ViT-L-14":      ("ViT-L-14",            "metaclip_fullcc"),
    "MetaCLIP-ViT-H-14":      ("ViT-H-14",            "metaclip_fullcc"),
    # SigLIP (sigmoid contrastive objective; zero-shot inference unchanged)
    "SigLIP-ViT-B-16":        ("ViT-B-16-SigLIP",     "webli"),
    "SigLIP-ViT-L-16-384":    ("ViT-L-16-384-SigLIP", "webli"),
}

# Original CLIP model strings (loaded via openai/CLIP library)
_ORIGINAL_CLIP = {
    "ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px",
    "RN50", "RN50x4", "RN50x16", "RN50x64", "RN101",
}


class VLMEncoder(torch.nn.Module):
    """
    Unified wrapper around any supported vision-language encoder.

    Always call self.tokenize(texts) rather than clip.tokenize() or
    open_clip.tokenize() — this ensures the correct vocabulary and
    context length for each backend.
    """

    def __init__(self, args, keep_lang: bool = True):
        super().__init__()
        self.model_name = args.model
        self.cache_dir  = getattr(args, "cache_dir", None)
        self._backend   = None   # "clip" | "openclip"
        self._tokenizer = None   # open_clip tokenizer (openclip backend only)

        if args.model in _ORIGINAL_CLIP:
            self._backend = "clip"
            device = getattr(args, "device", "cpu")
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                args.model, device, jit=False
            )
            print(f"[VLMEncoder] original CLIP: {args.model}")

        elif args.model in _OPENCLIP_REGISTRY:
            self._backend = "openclip"
            arch, pretrained = _OPENCLIP_REGISTRY[args.model]
            self.model, self.train_preprocess, self.val_preprocess = (
                open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            )
            self._tokenizer = open_clip.get_tokenizer(arch)
            print(f"[VLMEncoder] OpenCLIP: {arch} / {pretrained}")

        else:
            raise ValueError(
                f"Unknown model '{args.model}'.\n"
                f"Supported original CLIP: {sorted(_ORIGINAL_CLIP)}\n"
                f"Supported OpenCLIP/MetaCLIP/SigLIP: {sorted(_OPENCLIP_REGISTRY)}"
            )

    # ------------------------------------------------------------------
    # Tokenisation — always call this, never clip.tokenize() directly
    # ------------------------------------------------------------------

    def tokenize(self, texts, context_length: int = 77):
        """Return a LongTensor of token ids for any backend."""
        if self._backend == "clip":
            return clip.tokenize(texts, context_length=context_length)
        else:
            return self._tokenizer(texts)   # open_clip returns tensor directly

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def encode_image(self, images, normalize: bool = True):
        features = self.model.encode_image(images)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, token_ids, normalize: bool = True):
        features = self.model.encode_text(token_ids)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, images=None, text=None):
        if images is not None and text is None:
            return self.encode_image(images)
        elif images is None and text is not None:
            return self.encode_text(text)
        raise ValueError("Provide at least one of images or text.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename):
        print(f"Saving VLMEncoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading VLMEncoder from {filename}")
        return utils.torch_load(filename)


# Backward-compatibility alias
CLIPEncoder = VLMEncoder


# ------------------------------------------------------------------
# Classification head (unchanged from original)
# ------------------------------------------------------------------

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=(512, 1000)):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder     = image_encoder
        self.classification_head = classification_head
        self.process_images    = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess   = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        return self.classification_head(inputs)

    def save(self, filename):
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        return utils.torch_load(filename)
