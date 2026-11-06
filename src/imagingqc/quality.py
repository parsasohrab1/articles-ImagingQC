from __future__ import annotations

import numpy as np

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)


def _conv3(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1, mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i : i + img.shape[0], j : j + img.shape[1]]
    return out


def features(img: np.ndarray) -> np.ndarray:
    """Six quality features of an image scaled to [0, 1]: mean, std, 1st-99th percentile range,
    Laplacian variance (sharpness), robust noise estimate (MAD of the Laplacian), histogram entropy."""
    x = img.astype(float)
    lap = _conv3(x, LAP)
    hist, _ = np.histogram(x, bins=32, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    q1, q99 = np.percentile(x, [1, 99])
    mad = 1.4826 * np.median(np.abs(lap - np.median(lap)))
    return np.array(
        [x.mean(), x.std(), q99 - q1, np.log(lap.var() + 1e-9), np.log(mad + 1e-6), ent]
    )


def feature_matrix(imgs: np.ndarray) -> np.ndarray:
    return np.vstack([features(i) for i in imgs])


class MahalanobisQC:
    """One-class detector fitted on pristine images; threshold from a calibration set (conformal quantile)."""

    def fit(self, F: np.ndarray, ridge: float = 1e-3):
        self.mu = F.mean(0)
        C = np.cov(F.T) + ridge * np.eye(F.shape[1])
        self.P = np.linalg.inv(C)
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        d = F - self.mu
        return np.einsum("ij,jk,ik->i", d, self.P, d)

    def calibrate(self, F_cal: np.ndarray, alpha: float = 0.05):
        s = np.sort(self.score(F_cal))
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        self.thr = float(s[min(k, n) - 1])
        return self

    def reject(self, F: np.ndarray) -> np.ndarray:
        return self.score(F) > self.thr


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    p = np.pad(img, r, mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, p)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)


def degrade(img: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(float)
    if kind == "blur":
        y = gaussian_blur(x, level)
    elif kind == "noise":
        y = x + rng.normal(0, level, x.shape)
    elif kind == "low_contrast":
        y = x.mean() + level * (x - x.mean())
    elif kind == "brightness":
        y = x + level
    elif kind == "occlusion":
        y = x.copy()
        h = int(level * x.shape[0])
        r, c = rng.integers(0, x.shape[0] - h + 1, 2)
        y[r : r + h, c : c + h] = 0.0
    else:
        raise ValueError(kind)
    return np.clip(y, 0.0, 1.0)
