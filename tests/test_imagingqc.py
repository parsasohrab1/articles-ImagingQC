import numpy as np

from imagingqc.quality import MahalanobisQC, degrade, feature_matrix, features, gaussian_blur


def _imgs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    base = np.clip(0.5 + 0.2 * rng.standard_normal((n, 28, 28)), 0, 1)
    return np.array([gaussian_blur(b, 1.0) for b in base])


def test_blur_lowers_sharpness_and_noise_raises_noise_estimate():
    rng = np.random.default_rng(0)
    im = _imgs(1)[0]
    f0 = features(im)
    assert features(degrade(im, "blur", 2.0, rng))[3] < f0[3]
    assert features(degrade(im, "noise", 0.2, rng))[4] > f0[4]


def test_conformal_false_rejection_rate_on_pristine():
    I = _imgs(600)
    F = feature_matrix(I)
    qc = MahalanobisQC().fit(F[:200]).calibrate(F[200:400], 0.1)
    assert abs(qc.reject(F[400:]).mean() - 0.1) < 0.06


def test_detector_flags_strong_degradations():
    I = _imgs(400)
    F = feature_matrix(I)
    qc = MahalanobisQC().fit(F[:200]).calibrate(F[200:300], 0.05)
    rng = np.random.default_rng(1)
    bad = feature_matrix(np.array([degrade(i, "noise", 0.3, rng) for i in I[300:]]))
    assert qc.reject(bad).mean() > 0.9
