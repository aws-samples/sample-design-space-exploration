# Car Design Explorer Surrogate Models

## Overview

These trained PyTorch checkpoints support the Car Design Space Explorer educational sample. They provide rapid aerodynamic surrogate predictions for automotive design exploration and demonstrate an art of the possible workflow on AWS.

The models are not production CFD solvers and their outputs are not certified engineering results.

## Model artifacts

| Path | Purpose | Output |
|---|---|---|
| `backend/models/weights/kpi/best_model.pt` | KPI surrogate | Drag, side-force, lift, and yaw-moment coefficients, Cd, Cs, Cl, and Cmy |
| `backend/models/weights/surface/best_model.pt` | Surface-field surrogate | Average pressure and skin-friction coefficients, cpavg and cfxavg |
| `backend/models/weights/slices/ae_best_model.pt` | Slice autoencoder | Latent representation used for flow-field slice prediction |
| `backend/models/weights/slices/mgn_last_model.pt` | Slice MeshGraphNet | Velocity flow-field slice predictions |

The models are loaded through MLSimKit inference code in `backend/training/inference.py`.

## Training data

The models were trained using data derived from WindsorML, a high-fidelity computational fluid dynamics dataset for automotive aerodynamics. WindsorML contains geometric variants of the Windsor body with geometry, force and moment coefficients, and three-dimensional boundary and volume fields.

Dataset source: https://huggingface.co/datasets/neashton/windsorml

Dataset website: https://caemldatasets.org

Dataset license: Creative Commons Attribution-ShareAlike 4.0 International, CC BY-SA 4.0

Associated paper: https://arxiv.org/abs/2407.19320

Recommended citation:

Neil Ashton, Jordan Angel, Aditya Ghate, Gaetan Kenway, Man Long Wong, Cetin Kiris, Astrid Walle, Danielle Maddix, and Gary Page. WindsorML: High-Fidelity Computational Fluid Dynamics Dataset for Automotive Aerodynamics. 2024.

The repository does not currently contain a complete reproducibility manifest identifying the exact WindsorML revision, complete training split, hyperparameters, and evaluation run used for every checkpoint. Those details should be recorded before treating the models as reproducible research artifacts.

## Intended use

- Educational demonstrations of multi-agent automotive design exploration.
- Relative comparison of candidate geometries within the demonstrated design domain.
- Rapid prototyping of geometry to surrogate-analysis workflows.
- Non-production experimentation with Amazon Bedrock AgentCore and Strands Agents.

## Out-of-scope use

- Production vehicle design decisions.
- Safety-critical analysis or certification.
- Replacement for validated CFD, wind-tunnel testing, or professional engineering review.
- Claims of performance outside the WindsorML-derived training domain.
- Automated decisions where prediction errors could cause safety, legal, financial, or environmental harm.

## Limitations

- Predictions are surrogate estimates and may differ materially from CFD or physical testing.
- Accuracy can degrade for uploaded or generated geometries outside the training distribution.
- Model coverage and cached prediction coverage are partial and should not be interpreted as complete support for every advertised variant.
- Geometry orientation, scale, mesh quality, and topology can affect inference behavior.
- The checkpoints have not been independently validated for production use.
- No fairness, robustness, adversarial, or safety certification is provided.

## Evaluation

This repository is designed for comparative demonstrations rather than production benchmarking. Before external release, document per-checkpoint evaluation datasets, metrics, acceptance thresholds, and known failure cases.

## Licenses

The checkpoint files listed above are licensed under CC BY-SA 4.0. See `backend/models/weights/LICENSE`.

WindsorML retains its original CC BY-SA 4.0 license and attribution requirements.

Repository source code is licensed separately under the MIT No Attribution license. See `LICENSE`.

Third-party software remains under its respective license. See `THIRD_PARTY_NOTICES.md`.

## Disclaimer

These models are provided as an educational art of the possible sample. They have not been validated, tested, or supported for production use. Users are responsible for evaluating model suitability, accuracy, safety, licensing, and regulatory compliance for their intended application.
