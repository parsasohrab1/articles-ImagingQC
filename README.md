# articles-ImagingQC

Unsupervised quality control for chest X-ray images: a Mahalanobis detector on six image-quality features, fitted on pristine images only,
with a conformal threshold. Real images: PneumoniaMNIST (MedMNIST v2, 28x28). Degradations (blur, noise, contrast loss, brightness
shift, occlusion) are **injected synthetic perturbations** of the real test images — the data contain no labelled real quality problems. TRL 4.

```
pip install -e ".[dev]" && python -m pytest
# data: https://zenodo.org/records/10519652 (pneumoniamnist.npz; not committed)
python research/run_study.py <pneumoniamnist.npz>
```
