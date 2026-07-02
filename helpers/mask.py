import numpy as np
import cv2


def keep_largest_disc(mask):
    """Keeps only the largest connected component in the mask.
    Returns (cleaned_mask, changed, good)."""
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels <= 2:
        # Only background + one component -> nothing to clean
        return mask_bin, False, True

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    cleaned_mask = np.zeros_like(mask_bin)
    cleaned_mask[labels == largest_label] = 1
    cleaned_mask = mask * cleaned_mask

    cup = (cleaned_mask > 200).astype(np.uint8)
    disc = (cleaned_mask > 100).astype(np.uint8)
    good = True
    if disc.sum() > 0 and (cup.sum() / disc.sum() < 0.05):
        good = False

    return cleaned_mask, True, good


def evaluate_and_clean_mask(mask_path, disc_thresh=100, cup_thresh=200):
    """
    Loads an already-generated OD/OC mask (pixel values ~0 = background,
    ~127 = disc, ~255 = cup),
    cleans it (keeps only the largest connected disc component if there
    are stray blobs), and evaluates whether it passes quality checks
    (circularity, cup-to-disc ratio, cleaning outcome).

    Parameters
    ----------
    mask_path : str
        Path to the saved mask image.
    disc_thresh : int
        Pixel value threshold above which a pixel counts as disc.
    cup_thresh : int
        Pixel value threshold above which a pixel counts as cup.

    Returns
    -------
    cleaned_mask : np.ndarray (uint8)
        The final, cleaned mask.
    is_good : bool
        True if the mask passes the quality gate, False otherwise.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")

    d = (mask > disc_thresh).astype(np.uint8)
    c = (mask > cup_thresh).astype(np.uint8)

    # Cup-to-disc ratio
    disc_sum = d.sum()
    cup_sum = c.sum()
    cdr = cup_sum / disc_sum if disc_sum > 0 else 0

    # Circularity of the disc boundary
    contours, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

    # Clean the mask (keep only the largest connected disc component)
    cleaned_mask, changed, cleaning_good = keep_largest_disc(mask)

    is_good = (circularity > 0.6) and (cdr > 0.05) and (not changed or cleaning_good)

    final_mask = cleaned_mask if changed else mask

    return final_mask.astype(np.uint8), is_good


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, required=True, help="Path to a single mask image")
    parser.add_argument("--out", type=str, default="cleaned_mask.png", help="Where to save the cleaned mask")
    args = parser.parse_args()

    cleaned, good = evaluate_and_clean_mask(args.mask)
    if good:
        cv2.imwrite(args.out, cleaned)
        print(f"Cleaned mask saved to {args.out}")
        print(f"Good: {good}")