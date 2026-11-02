#!/usr/bin/env python3
"""PhotoGuard-style encoder attack on the Stable Diffusion VAE."""
import argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL


def to_t(img, dev):
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)


def to_img(t):
    a = t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((a * 255).round().astype(np.uint8))


def work_size(w, h, res):
    s = res / max(w, h)
    return max(8, round(w * s / 8) * 8), max(8, round(h * s / 8) * 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--eps", type=float, default=8, help="max change per pixel, /255 (higher = stronger, more visible)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--res", type=int, default=512, help="working resolution (lower if out of VRAM)")
    ap.add_argument("--target", default=None, help="optional target image (default: checkerboard)")
    a = ap.parse_args()

    out_path = a.output or a.input.rsplit(".", 1)[0] + "_protected.png"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"device: {dev}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to(dev).eval()
    vae.requires_grad_(False)

    orig_img = Image.open(a.input).convert("RGB")
    W, H = orig_img.size
    nw, nh = work_size(W, H, a.res)
    x = to_t(orig_img.resize((nw, nh), Image.BICUBIC), dev)

    # target image
    if a.target:
        tgt_img = to_t(Image.open(a.target).convert("RGB").resize((nw, nh)), dev)
    else:
        yy, xx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
        chk = (((yy // 32) + (xx // 32)) % 2).float()
        tgt_img = (chk[None, None].repeat(1, 3, 1, 1) * 0.8 + 0.1).to(dev)

    with torch.no_grad():
        tgt = vae.encode((tgt_img * 2 - 1).to(dtype)).latent_dist.mean.float()
        z0 = vae.encode((x * 2 - 1).to(dtype)).latent_dist.mean.float()
    print(f"latent distance before: {F.mse_loss(z0, tgt, reduction='sum').item():.0f}")

    eps = a.eps / 255.0
    alpha = eps / 16
    delta = ((torch.rand_like(x) * 2 - 1) * eps * 0.1).requires_grad_(True)

    for i in range(a.steps):
        adv = (x + delta).clamp(0, 1)
        z = vae.encode((adv * 2 - 1).to(dtype)).latent_dist.mean.float()
        loss = F.mse_loss(z, tgt, reduction="sum")  # sum avoids fp16 grad underflow
        loss.backward()
        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((x + delta).clamp(0, 1) - x)
        delta.grad = None
        if (i + 1) % 20 == 0 or i == 0:
            print(f"step {i+1}/{a.steps}  latent distance: {loss.item():.0f}")

    # apply perturbation to the full-res original
    orig_t = to_t(orig_img, dev)
    d = F.interpolate(delta.detach(), size=(H, W), mode="bicubic", align_corners=False).clamp(-eps, eps)
    prot = (orig_t + d).clamp(0, 1)
    out = to_img(prot)
    out.save(out_path, format="PNG")  # keep PNG, JPEG compression weakens the effect

    final = to_t(out, dev)
    mse = ((final - orig_t) ** 2).mean().item()
    psnr = 10 * math.log10(1 / max(mse, 1e-12))
    print(f"saved: {out_path}")
    print(f"max pixel change: {(final - orig_t).abs().max().item() * 255:.1f}/255   PSNR: {psnr:.1f} dB")


if __name__ == "__main__":
    main()
