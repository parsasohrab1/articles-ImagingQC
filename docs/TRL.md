# TRL assessment – ImagingQC

**Claimed level: TRL 4** (laboratory validation on public real images with injected degradations). Not TRL 5: 28x28 images, no real quality problems, no clinical setting.

Pristine test images (624): downstream accuracy 0.845; false rejection 0.018 (target 0.05; the threshold is conservative); accuracy on the accepted images 0.848.

| Injected degradation | Detection rate | Accepted share | Classifier accuracy without QC | Accuracy on accepted images |
|---|---|---|---|---|
| blur_0.7 | 0.538 | 0.462 | 0.832 | 0.767 |
| blur_1 | 0.989 | 0.011 | 0.817 | n/a (<20 accepted) |
| blur_1.5 | 1.000 | 0.000 | 0.776 | n/a (<20 accepted) |
| blur_2 | 1.000 | 0.000 | 0.716 | n/a (<20 accepted) |
| noise_0.03 | 0.287 | 0.713 | 0.851 | 0.838 |
| noise_0.06 | 0.997 | 0.003 | 0.845 | n/a (<20 accepted) |
| noise_0.1 | 1.000 | 0.000 | 0.838 | n/a (<20 accepted) |
| noise_0.2 | 1.000 | 0.000 | 0.801 | n/a (<20 accepted) |
| low_contrast_0.7 | 0.040 | 0.960 | 0.768 | 0.773 |
| low_contrast_0.5 | 0.224 | 0.776 | 0.681 | 0.645 |
| low_contrast_0.3 | 0.947 | 0.053 | 0.625 | 0.152 |
| brightness_0.1 | 0.030 | 0.970 | 0.841 | 0.841 |
| brightness_0.2 | 0.231 | 0.769 | 0.801 | 0.846 |
| brightness_0.3 | 0.753 | 0.247 | 0.736 | 0.864 |
| occlusion_0.2 | 0.583 | 0.417 | 0.793 | 0.696 |
| occlusion_0.4 | 0.997 | 0.003 | 0.676 | n/a (<20 accepted) |

## Findings and caveats
- The detector is very sensitive to blur and noise (blur sigma >= 1.0 and noise >= 0.06 are detected in more than 98 % of images) but weak for global intensity changes: contrast scaled to 0.7 is detected in 4 % of images and a brightness shift of 0.1 in 3 %, although contrast 0.7 already lowers the classifier accuracy from 0.845 to 0.768.
- **Rejection does not reliably help the classifier.** Accuracy on the accepted subset is often lower than without QC (blur 0.7: 0.767 against 0.832; contrast 0.5: 0.645 against 0.681; contrast 0.3: 0.152 on 5 % of images), because the images that escape detection are not a random sample: they are the ones whose degradation the detector cannot see, and for them the classifier may fail. Accepted-subset accuracy is only computed with at least 20 accepted images; small subsets are very noisy.
- The brightness shifts show the opposite pattern (accuracy on accepted images higher: 0.864 against 0.736 at +0.3): the detector removes the strongly brightened images that the classifier misreads.
- The downstream classifier is a weak baseline (accuracy 0.845 on pristine images), and the injected degradations are artificial; real acquisition problems (motion, positioning, exposure) differ.
- The conformal threshold guarantees the false-rejection rate only for images exchangeable with the validation set.
