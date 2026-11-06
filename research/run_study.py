"""PneumoniaMNIST (28x28 chest X-rays, MedMNIST v2, real images): unsupervised QC against INJECTED degradations.

usage: python research/run_study.py <pneumoniamnist.npz>
The detector is fitted only on pristine training images and calibrated (conformal quantile, alpha = 0.05) on pristine
validation images. Degradations (blur, noise, contrast loss, brightness shift, occlusion) are synthetic perturbations of the real
test images: the data contain no labelled real quality problems. A downstream pneumonia classifier shows what QC buys.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imagingqc.quality import MahalanobisQC, degrade, feature_matrix

warnings.filterwarnings("ignore")
ALPHA = 0.05
SETTINGS = {
    "blur": (0.7, 1.0, 1.5, 2.0),
    "noise": (0.03, 0.06, 0.1, 0.2),
    "low_contrast": (0.7, 0.5, 0.3),
    "brightness": (0.1, 0.2, 0.3),
    "occlusion": (0.2, 0.4),
}


def main(path: str):
    d = np.load(path)
    Xtr, ytr = d["train_images"] / 255.0, d["train_labels"].ravel()
    Xva = d["val_images"] / 255.0
    Xte, yte = d["test_images"] / 255.0, d["test_labels"].ravel()
    qc = MahalanobisQC().fit(feature_matrix(Xtr)).calibrate(feature_matrix(Xva), ALPHA)
    clf = make_pipeline(
        StandardScaler(), PCA(50, random_state=0), LogisticRegression(max_iter=2000)
    )
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    flat = lambda A: A.reshape(len(A), -1)
    base_acc = float(clf.score(flat(Xte), yte))
    rej0 = qc.reject(feature_matrix(Xte))
    out = {
        "note": "real PneumoniaMNIST images; degradations are INJECTED; detector fitted on pristine train images",
        "n_test": len(yte),
        "classifier_accuracy_pristine": base_acc,
        "false_rejection_pristine_test": float(rej0.mean()),
        "accuracy_on_accepted_pristine": float(clf.score(flat(Xte[~rej0]), yte[~rej0])),
        "settings": {},
    }
    rng = np.random.default_rng(0)
    for kind, levels in SETTINGS.items():
        for lv in levels:
            D = np.array([degrade(im, kind, lv, rng) for im in Xte])
            rej = qc.reject(feature_matrix(D))
            acc_all = float(clf.score(flat(D), yte))
            acc_acc = float(clf.score(flat(D[~rej]), yte[~rej])) if (~rej).sum() > 20 else None
            out["settings"][f"{kind}_{lv:g}"] = {
                "detection_rate": float(rej.mean()),
                "classifier_accuracy_no_qc": acc_all,
                "classifier_accuracy_on_accepted": acc_acc,
                "accepted_share": float((~rej).mean()),
            }
            print(
                kind,
                lv,
                {
                    k: (None if v is None else round(v, 3))
                    for k, v in out["settings"][f"{kind}_{lv:g}"].items()
                },
            )
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))
    print(
        "pristine acc",
        round(base_acc, 3),
        "false rej",
        round(out["false_rejection_pristine_test"], 3),
    )


if __name__ == "__main__":
    main(sys.argv[1])
