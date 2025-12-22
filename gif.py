import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import (
    ImageNormalize, PercentileInterval, AsinhStretch, LogStretch, SqrtStretch
)
from astropy.stats import sigma_clip

# Optional (for GIF)
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


# -----------------------------
# Helpers
# -----------------------------
def load_fits_primary(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError("No image data in primary HDU (hdul[0].data is None)")
    return np.asarray(data)


def squeeze_to_stack(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Returns:
      stack: shape (nplanes, ny, nx)
      (ny, nx)
    """
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        raise ValueError(f"Data has ndim={arr.ndim}, expected at least 2D")

    if arr.ndim == 2:
        ny, nx = arr.shape
        stack = arr.reshape(1, ny, nx)
        return stack, (ny, nx)

    ny, nx = arr.shape[-2], arr.shape[-1]
    stack = arr.reshape(-1, ny, nx)
    return stack, (ny, nx)


def sanitize_finite(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64, copy=False)
    finite = np.isfinite(img)
    fill = np.median(img[finite]) if finite.any() else 0.0
    return np.where(finite, img, fill)


def preprocess(
    img: np.ndarray,
    do_sigma_clip: bool = True,
    clip_sigma: float = 3.5,
    clip_iters: int = 5,
    subtract_background: bool = True,
    background_percentile: float = 20.0,
) -> np.ndarray:
    """
    - sigma-clip (optional) to reduce hot pixels / cosmic rays
    - background subtraction using a low percentile estimate
    """
    img = sanitize_finite(img)

    if do_sigma_clip:
        clipped = sigma_clip(img, sigma=clip_sigma, maxiters=clip_iters)
        # Replace masked values with median of unmasked
        if np.ma.isMaskedArray(clipped) and np.any(clipped.mask):
            med = np.ma.median(clipped)
            img = clipped.filled(float(med))
        else:
            img = np.asarray(clipped)

    if subtract_background:
        bg = np.percentile(img, background_percentile)
        img = img - bg

    return img


def make_norm(img: np.ndarray, mode: str = "asinh", p: float = 99.5):
    interval = PercentileInterval(p)
    stretch = {
        "asinh": AsinhStretch(),
        "log": LogStretch(),
        "sqrt": SqrtStretch(),
        "linear": None,
    }.get(mode, AsinhStretch())

    if stretch is None:
        return ImageNormalize(img, interval=interval)
    return ImageNormalize(img, interval=interval, stretch=stretch)


def save_png(img, out_png, cmap="gray", norm=None, dpi=250, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.imshow(img, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def to_uint8(img: np.ndarray, norm: ImageNormalize) -> np.ndarray:
    # Map image through norm to [0,1], then to [0,255]
    x = norm(img)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def make_rgb_from_three_planes(r, g, b, norm_mode="asinh", p=99.7) -> np.ndarray:
    nr = make_norm(r, mode=norm_mode, p=p)
    ng = make_norm(g, mode=norm_mode, p=p)
    nb = make_norm(b, mode=norm_mode, p=p)
    R = to_uint8(r, nr)
    G = to_uint8(g, ng)
    B = to_uint8(b, nb)
    return np.dstack([R, G, B])  # uint8 RGB


# -----------------------------
# Main
# -----------------------------
fits_path = "ngc6503.fits"
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

data = load_fits_primary(fits_path)
print("Raw shape:", data.shape)

stack, (ny, nx) = squeeze_to_stack(data)
nplanes = stack.shape[0]
print("Stack shape:", stack.shape, "(nplanes, ny, nx)")

# Choose a single plane to render (0..nplanes-1)
k = 0
img = preprocess(stack[k], do_sigma_clip=True, subtract_background=True)

# ---- Grayscale HD ----
norm_gray = make_norm(img, mode="asinh", p=99.5)
save_png(img, os.path.join(out_dir, f"plane_{k:03d}_gray_hd.png"),
         cmap="gray", norm=norm_gray, dpi=300, figsize=(8, 8))

# ---- False color "pretty" versions ----
# Try a few colormaps; pick the one you like
for cmap in ["magma", "inferno", "viridis", "turbo"]:
    save_png(img, os.path.join(out_dir, f"plane_{k:03d}_{cmap}_hd.png"),
             cmap=cmap, norm=norm_gray, dpi=300, figsize=(8, 8))

# ---- Optional RGB composite from 3 different planes ----
# If you have at least 3 planes, you can map them to R/G/B for a "real-ish" composite.
# You can change these indices to something meaningful for your dataset.
if nplanes >= 3:
    r_i, g_i, b_i = 0, min(1, nplanes - 1), min(2, nplanes - 1)
    r = preprocess(stack[r_i], do_sigma_clip=True, subtract_background=True)
    g = preprocess(stack[g_i], do_sigma_clip=True, subtract_background=True)
    b = preprocess(stack[b_i], do_sigma_clip=True, subtract_background=True)

    rgb = make_rgb_from_three_planes(r, g, b, norm_mode="asinh", p=99.7)

    plt.imsave(os.path.join(out_dir, f"rgb_{r_i:03d}_{g_i:03d}_{b_i:03d}.png"), rgb)
    print("Saved RGB composite PNG")

# ---- Animated GIF through planes ----
# Makes a GIF by rendering each plane with a consistent normalization.
# Note: for a single 2D image (nplanes=1), this will just make a 1-frame GIF.
if HAS_IMAGEIO:
    gif_path = os.path.join(out_dir, "cube_animation.gif")

    # Preprocess all planes first (can be heavy for large cubes)
    proc = np.stack([preprocess(stack[i], do_sigma_clip=True, subtract_background=True)
                     for i in range(nplanes)], axis=0)

    # Use a GLOBAL norm so brightness doesn't "pump" between frames
    global_norm = make_norm(proc, mode="asinh", p=99.7)

    frames = []
    for i in range(nplanes):
        frame = to_uint8(proc[i], global_norm)
        frames.append(frame)

    # duration in seconds per frame
    imageio.mimsave(gif_path, frames, duration=0.08)
    print("Saved GIF:", gif_path)
else:
    print("imageio not installed; skipping GIF. Install with: pip install imageio")

print("All outputs saved under:", out_dir)
