import os
import cv2
import numpy as np
from pathlib import Path


def circular_crop(image_path: str, output_dir: str, padding: int = 0):
    """
    Detect the fundus circle, mask everything outside it to black,
    and save a tight square crop around it.

    Parameters
    ----------
    image_path  : path to the input fundus image
    output_dir  : directory to save the cropped image
    padding     : extra pixels to expand the detected circle radius (default 0)
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

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(image_path).stem
    out_path = os.path.join(output_dir, stem + "_cropped.jpg")
    cv2.imwrite(out_path, cropped)
    print(f"Saved → {out_path}  (centre=({cx},{cy}), r={r})")

    return cropped


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Circular crop a fundus image")
    p.add_argument("--image_path",  type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--padding",     type=int, default=0,
                   help="Extra pixels added to detected radius (default 0)")
    args = p.parse_args()

    circular_crop(args.image_path, args.output_dir, args.padding)