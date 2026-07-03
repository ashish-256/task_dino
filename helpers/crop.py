import os
import cv2
import numpy as np
from pathlib import Path


def circular_crop(image_path: str, output_dir: str, padding: int = 0,
                   resize_to: int = 1024):
    """
    Detect the fundus circle, mask everything outside it to black,
    save a tight square crop around it, and resize to a fixed size.

    Parameters
    ----------
    image_path  : path to the input fundus image
    output_dir  : directory to save the cropped image
    padding     : extra pixels to expand the detected circle radius (default 0)
    resize_to   : output side length in pixels for the final square image
                  (default 1024). Set to None to skip resizing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ── Detect the fundus circle ──────────────────────────────────────────────
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Largest contour = fundus boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No fundus region found — image may be fully black.")

    largest     = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    cx, cy, r   = int(cx), int(cy), int(r) + padding

    # ── Apply circular mask ───────────────────────────────────────────────────
    h, w      = img.shape[:2]
    mask      = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    masked    = cv2.bitwise_and(img, img, mask=mask)

    # ── Tight square crop around the circle ──────────────────────────────────
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, w)
    y2 = min(cy + r, h)
    cropped = masked[y1:y2, x1:x2]

    # ── Resize to fixed output size ───────────────────────────────────────────
    if resize_to is not None:
        current_side = cropped.shape[0]
        interp = cv2.INTER_AREA if current_side > resize_to else cv2.INTER_CUBIC
        cropped = cv2.resize(cropped, (resize_to, resize_to), interpolation=interp)

    # ── Save with the original filename, in the output directory ─────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(out_path, cropped)
    print(f"Saved → {out_path}  (centre=({cx},{cy}), r={r}, size={cropped.shape[1]}x{cropped.shape[0]})")

    return cropped


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Circular crop a fundus image")
    p.add_argument("--image_path",  type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--padding",     type=int, default=0,
                   help="Extra pixels added to detected radius (default 0)")
    p.add_argument("--resize_to",   type=int, default=1024,
                   help="Output square side length in pixels (default 1024, use 0 to skip resizing)")
    args = p.parse_args()

    resize_to = None if args.resize_to == 0 else args.resize_to
    circular_crop(args.image_path, args.output_dir, args.padding, resize_to)