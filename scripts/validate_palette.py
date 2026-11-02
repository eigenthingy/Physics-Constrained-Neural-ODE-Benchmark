"""Colour-palette validator (OKLab ΔE + dichromat simulation + WCAG contrast).

Kept in the repo so the figure palette is a *checked* decision, not a taste call.

    python scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --mode light

Gates (OKLab ΔE x100):
  * normal-vision ΔE >= 15 for every pair        (hard floor)
  * simulated-CVD  ΔE >= 8  for every pair       (target; 6-8 needs 2nd encoding)
  * contrast vs surface >= 3:1                   (warn -> requires direct labels)

References: Viénot, Brightwell & Mollon (1999) dichromat simulation;
Ottosson (2020) OKLab.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

# --------------------------------------------------------------------------- #
# colour-space plumbing
# --------------------------------------------------------------------------- #

_LMS = np.array(
    [
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709],
    ]
)
_LMS_INV = np.linalg.inv(_LMS)

_CVD = {
    "protanopia": np.array([[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "deuteranopia": np.array([[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827], [0.0, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.395913, 0.801109, 0.0]]),
}

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = _M1 @ srgb_to_linear(rgb)
    return _M2 @ np.cbrt(lms)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    lms = _LMS @ lin
    return linear_to_srgb(_LMS_INV @ (_CVD[kind] @ lms))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(to_oklab(a) - to_oklab(b)) * 100.0)


def luminance(rgb: np.ndarray) -> float:
    return float(np.dot(srgb_to_linear(rgb), [0.2126, 0.7152, 0.0722]))


def contrast(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def validate(hexes: list[str], surface: str) -> bool:
    cols = [hex_to_rgb(h) for h in hexes]
    surf = hex_to_rgb(surface)
    ok = True

    print(f"{'pair':<20} {'normal':>8} {'prot':>8} {'deut':>8} {'trit':>8}  verdict")
    print("-" * 66)
    for i, j in itertools.combinations(range(len(cols)), 2):
        normal = delta_e(cols[i], cols[j])
        sims = {k: delta_e(simulate(cols[i], k), simulate(cols[j], k)) for k in _CVD}
        worst = min(sims.values())
        if normal < 15:
            verdict, ok = "FAIL normal<15", False
        elif worst < 6:
            verdict, ok = "FAIL cvd<6", False
        elif worst < 8:
            verdict = "WARN cvd 6-8"
        else:
            verdict = "pass"
        print(
            f"{hexes[i]}/{hexes[j]:<11} {normal:8.1f} "
            f"{sims['protanopia']:8.1f} {sims['deuteranopia']:8.1f} "
            f"{sims['tritanopia']:8.1f}  {verdict}"
        )

    print()
    for h, c in zip(hexes, cols):
        cr = contrast(c, surf)
        tag = "pass" if cr >= 3.0 else "WARN (needs direct label / table)"
        print(f"contrast {h} vs {surface}: {cr:5.2f}:1  {tag}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface", default=None)
    a = ap.parse_args()
    surface = a.surface or ("#fcfcfb" if a.mode == "light" else "#1a1a19")
    good = validate([h for h in a.palette.split(",") if h.strip()], surface)
    print("\nRESULT:", "PASS" if good else "FAIL")
