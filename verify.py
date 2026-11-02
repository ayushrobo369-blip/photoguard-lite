import sys, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL

orig_p, prot_p = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dt).to(dev).eval()

def roundtrip(path, size):
    img = Image.open(path).convert("RGB").resize(size, Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        z = vae.encode((x * 2 - 1).to(dt)).latent_dist.mean
        y = vae.decode(z).sample
    y = ((y.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return np.asarray(img), y

W, H = Image.open(orig_p).size
s = 512 / max(W, H)
size = (round(W * s / 8) * 8, round(H * s / 8) * 8)
a_in, a_out = roundtrip(orig_p, size)
b_in, b_out = roundtrip(prot_p, size)
grid = np.concatenate([np.concatenate([a_in, a_out], 1), np.concatenate([b_in, b_out], 1)], 0)
Image.fromarray(grid).save("verify.png")
print("saved verify.png  (top: original in/out, bottom: protected in/out)")
