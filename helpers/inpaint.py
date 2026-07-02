import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from simple_lama_inpainting import SimpleLama
from simple_lama_inpainting.utils.util import prepare_img_and_mask
from scipy.ndimage import gaussian_filter, distance_transform_edt


def run_lama(
    fundus_path: str,
    mask_raw: np.ndarray,
    scale_factor: float = None,
    pad_modulo: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    feather_px: float = 8.0,
    colour_correct: bool = True,
    context_ring_px: int = 25,
    mask_dilation: int = 5,
):
    img_bgr  = cv2.imread(fundus_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil  = Image.fromarray(img_rgb)

    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read mask: {fundus_path}")

    mask_uint8 = (mask_raw * 255).astype(np.uint8)
    _, mask_bin = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask_bin > 0).mean() > 0.5:
        mask_bin = cv2.bitwise_not(mask_bin)

    if mask_bin.shape[:2] != img_bgr.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    if mask_dilation > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilation+1, 2*mask_dilation+1))
        mask_bin = cv2.dilate(mask_bin, dk)

    mask_pil  = Image.fromarray(mask_bin).convert("L")
    mask_bool = mask_bin > 0
    img_arr   = np.array(img_pil)
    mask_arr  = np.array(mask_pil)

    img_t, mask_t = prepare_img_and_mask(
        img_arr, mask_arr,
        device=torch.device(device),
        pad_out_to_modulo=pad_modulo,
        scale_factor=scale_factor,
    )

    lama = SimpleLama(device=torch.device(device))

    with torch.inference_mode():
        out    = lama.model(img_t, mask_t)
        result = out[0].permute(1,2,0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

    if scale_factor is not None and scale_factor != 1.0:
        h, w   = img_bgr.shape[:2]
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        h, w   = img_bgr.shape[:2]
        result = result[:h, :w]

    if feather_px > 0:
        dist_in  = distance_transform_edt( mask_bool).astype(np.float32)
        dist_out = distance_transform_edt(~mask_bool).astype(np.float32)
        alpha    = np.where(mask_bool,
                            np.clip(dist_in  / feather_px, 0, 1),
                            1.0 - np.clip(dist_out / feather_px, 0, 1))
        alpha    = gaussian_filter(alpha, sigma=feather_px * 0.4)[..., np.newaxis]
        blended  = result.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1 - alpha)
        result   = np.clip(blended, 0, 255).astype(np.uint8)

    if colour_correct:
        ring_k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                          (2*context_ring_px+1, 2*context_ring_px+1))
        context_ring = cv2.dilate(mask_bin, ring_k) & (~mask_bool).astype(np.uint8)*255
        result_cc   = result.astype(np.float32)
        for ch in range(3):
            if context_ring.sum() > 100:
                ctx_mean = img_rgb[:,:,ch][context_ring > 0].mean()
                inp_mean = result_cc[:,:,ch][mask_bool].mean()
                shift    = (ctx_mean - inp_mean) * 0.4
                result_cc[:,:,ch][mask_bool] += shift
        result = np.clip(result_cc, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LaMa vessel inpainting — single image")
    p.add_argument("--image_path",        type=str, required=True)
    p.add_argument("--mask_path",         type=str, required=True)
    p.add_argument("--output_dir",        type=str, required=True)
    p.add_argument("--scale-factor",      type=float, default=None)
    p.add_argument("--pad-modulo",        type=int,   default=8)
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--feather",           type=float, default=8.0)
    p.add_argument("--no-colour-correct", action="store_true")
    p.add_argument("--context-ring",      type=int,   default=25)
    p.add_argument("--mask-dilation",     type=int,   default=5)
    args = p.parse_args()

    mask_raw = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask_path}")

    result_bgr = run_lama(
        fundus_path     = args.image_path,
        mask_raw        = mask_raw,
        scale_factor    = args.scale_factor,
        pad_modulo      = args.pad_modulo,
        device          = args.device,
        feather_px      = args.feather,
        colour_correct  = not args.no_colour_correct,
        context_ring_px = args.context_ring,
        mask_dilation   = args.mask_dilation,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(args.image_path).stem
    out_path = os.path.join(args.output_dir, stem + "_paint.jpg")
    cv2.imwrite(out_path, result_bgr)
    print(f"Saved → {out_path}")