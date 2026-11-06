# Software Requirements Specification – ImagingQC

- SR-1 `quality.features`: mean, standard deviation, 1st–99th percentile range, log Laplacian variance (sharpness), log robust noise estimate (MAD of the Laplacian), histogram entropy.
- SR-2 `quality.MahalanobisQC`: fit mean/covariance on pristine training images; `calibrate`: conformal quantile (alpha = 0.05) of Mahalanobis distances on pristine validation images; `reject`: distance above threshold.
- SR-3 `quality.degrade`: blur (Gaussian), noise (additive Gaussian), low_contrast (scaling about the mean), brightness (additive shift), occlusion (black square), each clipped to [0, 1].
- SR-4 `research/run_study.py`: PneumoniaMNIST train/val/test split as released; downstream classifier = standardisation + PCA(50) + logistic regression trained on pristine images; detection rate, accepted share and classifier accuracy per setting.
- SR-5 Three unit tests.
