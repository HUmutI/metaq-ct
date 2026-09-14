"""Minimal stand-in for CT-CLIP's transformer_maskgit package.

MPS-CT imports exactly one symbol from it -- get_optimizer, in CTCLIPTrainer.py:5
and zero_shot.py:3.  The upstream package's __init__ pulls in MaskGITTransformer
-> ctvit -> vector_quantize_pytorch, ctvit_trainer -> ema_pytorch, and data ->
cv2, none of which MPS-CT touches.  Installing that chain into the env would add
opencv and a VQ-GAN stack to satisfy one AdamW wrapper.

optimizer.py is copied verbatim from
  BENCHMARK/CT-CLIP/transformer_maskgit/transformer_maskgit/optimizer.py
and imports only torch.optim, so behaviour is identical.
"""
