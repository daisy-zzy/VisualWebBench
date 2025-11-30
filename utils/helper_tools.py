import os
from typing import Optional, List, Dict, Any

import torch
from PIL import Image

import cv2
import numpy as np

# =================== Shared helpers =================== #

def crop_normalized(img: Image.Image, bbox):
    """
    bbox: [x0, y0, x1, y1] in normalized coordinates [0, 1].
    Returns a cropped image, resized back to the original size.

    NOTE: With a regular k×k grid plus symmetric margin, the crop
    keeps (approximately) the same aspect ratio as the original,
    so resizing back to (W, H) is fine.
    """
    w, h = img.size
    x0, y0, x1, y1 = bbox

    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))

    left = int(x0 * w)
    top = int(y0 * h)
    right = int(x1 * w)
    bottom = int(y1 * h)

    if right <= left or bottom <= top:
        # Degenerate box → return original image
        return img

    cropped = img.crop((left, top, right, bottom))
    # Resize back to original resolution to keep text legible
    return cropped.resize((w, h), Image.BICUBIC)


def build_zone_mapping(grid_size: int):
    """
    Build:
      - zone_to_bbox: mapping from zone name → [x0,y0,x1,y1]
      - synonyms: optional extra names that alias canonical zones (for k=3).

    Canonical names are always 'R<i>C<j>'.

    For k = 3, we also accept TL/TM/... as synonyms mapping to R1C1...R3C3.
    """
    k = grid_size
    zone_to_bbox: Dict[str, List[float]] = {}
    synonyms: Dict[str, str] = {}

    cell_w = 1.0 / k
    cell_h = 1.0 / k

    for row in range(1, k + 1):
        for col in range(1, k + 1):
            x0 = (col - 1) * cell_w
            x1 = col * cell_w
            y0 = (row - 1) * cell_h
            y1 = row * cell_h
            name = f"R{row}C{col}"
            zone_to_bbox[name] = [x0, y0, x1, y1]

    # Add TL/TM/... synonyms only when k = 3
    if k == 3:
        synonyms = {
            "TL": "R1C1",
            "TM": "R1C2",
            "TR": "R1C3",
            "ML": "R2C1",
            "MM": "R2C2",
            "MR": "R2C3",
            "BL": "R3C1",
            "BM": "R3C2",
            "BR": "R3C3",
        }

    return zone_to_bbox, synonyms


# ---------- Heading OCR helpers ---------- #

def clean_heading(raw: str) -> str:
    """
    Heuristic cleaner for heading OCR.

    - Strips quotes and whitespace.
    - If there's a colon, prefers the part after the colon
      (e.g., 'The main heading is: Hello World' -> 'Hello World').
    """
    if raw is None:
        return ""

    text = raw.strip().strip('"').strip("'")

    # If model output is like "The main heading is: XYZ"
    if ":" in text:
        before, after = text.split(":", 1)
        after = after.strip()
        if after:  # only use if non-empty
            text = after

    return text.strip()


def extract_top_heading_from_summary(summary_text: str) -> str:
    """
    Given the heading-summary output (list of headings),
    extract the first heading candidate.

    Expected format (but we stay lenient):
      1. Some Heading Text
      2. Another Heading

    We take the first non-empty line, strip leading '1.' etc.
    """
    if not summary_text:
        return ""

    lines = [ln.strip() for ln in summary_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]

    # Remove leading numbering like '1.' or '1)'
    # e.g. "1. Hello World" -> "Hello World"
    if first[0].isdigit():
        # find first space or dot after digits
        i = 0
        while i < len(first) and first[i].isdigit():
            i += 1
        # skip dot or ) and following spaces
        while i < len(first) and first[i] in [".", ")", " "]:
            i += 1
        first = first[i:].strip()

    return clean_heading(first)