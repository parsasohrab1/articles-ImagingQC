# Business Requirements Specification – ImagingQC

| ID | Requirement |
|----|-------------|
| BR-1 | Reject images of insufficient quality before they reach a diagnostic model, with a controlled false-rejection rate on good images. |
| BR-2 | Work without examples of bad images (fit on good images only). |
| BR-3 | Report, per degradation type and severity, the detection rate and what rejection does to the downstream classifier's accuracy. |
| BR-4 | State that degradations are injected and the images are 28x28; no clinical claim. |

Out of scope: full-resolution radiographs, real acquisition artefacts, clinical workflow integration.
