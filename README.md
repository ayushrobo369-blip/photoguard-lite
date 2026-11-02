# photoguard-lite

PhotoGuard-style protection against AI image editing. Adds a small, texture-adaptive
perturbation that confuses the Stable Diffusion VAE encoder.

## Install
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

## Use
    python photoguard_lite.py photo.jpg --eps 12 --floor 0.1
    python verify.py photo.jpg photo_protected.png

Share the output as PNG; JPEG compression weakens the protection.
Only targets SD 1.x VAE. A deterrent, not a guarantee.
