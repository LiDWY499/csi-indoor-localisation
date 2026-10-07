# Wi-Fi CSI Indoor Localisation

Comparing classical machine learning and deep learning for indoor positioning—and testing what happens when transmitter orientation changes.

Final-year BSc Computer Science and Artificial Intelligence project at Queen Mary University of London, by Derek Li (2026).

## The problem

GPS is unreliable indoors. Wi-Fi Channel State Information (CSI) captures signal characteristics across antenna links and subcarriers that can act as location fingerprints. This project predicts reference-point locations from CSI amplitude measurements.

## What the project includes

- Python pipelines for MATLAB data conversion, amplitude processing, numerical cleaning and normalisation.
- Packet-window classification with k-nearest neighbours, SVM and Random Forest baselines.
- DeepFi-style, CNN, TCN and Transformer implementations using PyTorch.
- Evaluation using localisation error and threshold-based metrics.
- A separate orientation-transfer experiment to probe distribution shift.

## Reported results

These figures come from the submitted project report; they have not been independently reproduced in this portfolio package.

| Experiment | Model | Mean localisation error |
| --- | --- | --- |
| Qiang Lab, matched conditions | Tuned Transformer | 0.109 m |
| Qiang Meeting Room, matched conditions | Tuned Transformer | 0.040 m |
| Figshare, 30° training → −60° testing | Transformer | 1.686 m |

**The key finding:** strong performance on matched train/test conditions did not establish robustness to a new transmitter orientation.

This is reference-point classification followed by coordinate mapping, not evidence of centimetre-level continuous positioning in an unseen building. Correct classifications produce exactly zero coordinate error. Mixed-orientation results and orientation-transfer results answer different questions and should not be combined into one accuracy claim.

## Pipeline

```mermaid
flowchart LR
    A[Public CSI datasets] --> B[Convert and clean amplitudes]
    B --> C[Prepare train/test data]
    C --> D[Packet windows]
    D --> E[Classical and deep models]
    E --> F[Reference-point prediction]
    F --> G[Coordinates and error metrics]
    G --> H[Compare matched and shifted conditions]
```

## Getting started

Use an isolated Python environment and install the dependencies:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

See [experiment instructions](docs/EXPERIMENTS.md) for dataset acquisition, preprocessing, training and evaluation commands. Run those commands from the repository root. The original instructions use Windows paths; adapt separators on other platforms.

Datasets, trained weights and generated results are not bundled. Full experiments require downloading the referenced datasets and may benefit from a GPU. Dependency versions and a tested Python version have not yet been recorded; installation and end-to-end execution remain to be verified.

## Repository structure

- `src/`: data preparation, baselines and neural model scripts.
- `docs/EXPERIMENTS.md`: original experimental workflow, with one broken command line repaired.
- `requirements.txt`: original dependency list.

## Limitations and next steps

- Validate generalisation using capture/session-separated splits and unseen environments.
- Record dependency versions and add small preprocessing/evaluation tests.
- Preserve raw run metrics and seed/configuration records alongside future reproduced results.
- Investigate orientation shift and domain adaptation before making deployment claims.

## Attribution and provenance

The experiments use the Qiang CSI and Figshare Intel 5300 datasets referenced in the experiment guide. Dataset ownership and reuse conditions remain with their authors; datasets are not redistributed here. The DeepFi-style implementation should not be presented as the original authors’ implementation.

The submitted report acknowledges ChatGPT assistance with language refinement, discussion of Transformer hyperparameters and regularisation, and a progress bar. This portfolio README was also prepared with AI assistance. Reported findings remain attributed to the submitted report.
