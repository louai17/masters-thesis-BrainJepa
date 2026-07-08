---
license: apache-2.0
language:
- en
tags:
- vision-language
- image-text
- vit
- gpt2
- self-supervised-learning
- jepa
- non-contrastive
---

# LeVLJEPA — vit_base_patch16_224 / DataComp-large

Checkpoint for **LeVLJEPA**, a non-contrastive vision-language pretraining method based on joint-embedding prediction and SIGReg regularisation (no negatives, no temperature, no momentum encoder).
Trained on DataComp-large for 200,000 steps with batch size 4,096.

## Model summary

| Property | Value |
|---|---|
| Method | LeVLJEPA |
| Vision encoder | `vit_base_patch16_224` (timm) |
| Text encoder | GPT-2 (12L / 12H / 768D) |
| Embedding dim | 256 |
| Projection head | Linear→BN→GELU→Linear (width 2048) |
| Training objective | Cross-modal prediction + SIGReg |
| Training data | DataComp-large |
| Training steps | 200,000 |

## Usage

```python
import torch
import timm
from transformers import GPT2Config, GPT2Model, AutoTokenizer
from safetensors.torch import load_file
from torchvision.ops import MLP
import torch.nn as nn

HIDDEN = 768
EMBED  = 256

# ── Vision encoder ───────────────────────────────────────────────────────────
vision_encoder = timm.create_model(
    "vit_base_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
)
vision_pre_proj = nn.Sequential(
    nn.Linear(HIDDEN, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, EMBED)
)

# ── Text encoder ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_with_eos_readout(tokenizer, text, max_length=77):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )["input_ids"] + [tokenizer.eos_token_id]
    pad_len = max_length - len(ids)
    input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * pad_len])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * pad_len])
    return dict(input_ids=input_ids, attention_mask=attention_mask)

def last_unmasked_token(hidden, attention_mask):
    lengths = attention_mask.sum(dim=1).clamp(min=1).long()
    gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather_idx).squeeze(1)

text_encoder = GPT2Model(GPT2Config(
    n_embd=HIDDEN, n_layer=12, n_head=12,
    n_inner=HIDDEN * 4, vocab_size=tokenizer.vocab_size,
    attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
))
text_pre_proj = nn.Sequential(
    nn.Linear(HIDDEN, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, EMBED)
)

# ── Load weights ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download

vision_weights = load_file(hf_hub_download("lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k", "vision_encoder.safetensors"))
text_weights   = load_file(hf_hub_download("lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k", "text_encoder.safetensors"))

vision_encoder.load_state_dict({k[len("encoder."):]: v for k, v in vision_weights.items() if k.startswith("encoder.")})
vision_pre_proj.load_state_dict({k[len("pre_proj."):]: v for k, v in vision_weights.items() if k.startswith("pre_proj.")})
text_encoder.load_state_dict({k[len("encoder."):]: v for k, v in text_weights.items() if k.startswith("encoder.")})
text_pre_proj.load_state_dict({k[len("pre_proj."):]: v for k, v in text_weights.items() if k.startswith("pre_proj.")})

vision_encoder.eval()
text_encoder.eval()

# ── Encode an image ──────────────────────────────────────────────────────────
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("image.jpg").convert("RGB")
pixel_values = transform(image).unsqueeze(0)

with torch.no_grad():
    image_features = vision_pre_proj(vision_encoder(pixel_values))  # (1, 256)

# ── Encode a caption ─────────────────────────────────────────────────────────
inputs = tokenize_with_eos_readout(tokenizer, "a photo of a cat")
with torch.no_grad():
    hidden = text_encoder(**inputs).last_hidden_state
    text_hidden = last_unmasked_token(hidden, inputs["attention_mask"])
    text_features = text_pre_proj(text_hidden)  # (1, 256)
```

## Files

| File | Contents |
|---|---|
| `vision_encoder.safetensors` | Vision encoder (`encoder.*`), pre-projection head (`pre_proj.*`), and cross-modal projector MLP (`projector.*`) |
| `text_encoder.safetensors`   | Text encoder (`encoder.*`), pre-projection head (`pre_proj.*`), and cross-modal projector MLP (`projector.*`) |
| `config.json` | Architecture and training hyperparameters |
