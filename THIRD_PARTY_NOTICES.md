# Third-Party Notices

This repository uses and, in some deployment paths, redistributes third-party software and data. Those components remain subject to their original license terms.

## MLSimKit

Component: MLSimKit

Recorded version: `0.1.1.dev12+g398db498c`

Source project: https://github.com/awslabs/ai-surrogate-models-in-engineering-on-aws

License: Apache License 2.0

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

The bundled distribution at `backend/mlsimkit_dist/mlsimkit_package.tar.gz` contains the upstream license, notice, and package metadata. Those files must remain with redistributed copies of the package.

## WindsorML

Component: WindsorML dataset

Source: https://huggingface.co/datasets/neashton/windsorml

Paper: https://arxiv.org/abs/2407.19320

License: Creative Commons Attribution-ShareAlike 4.0 International, CC BY-SA 4.0

WindsorML attribution and model-training details are documented in `MODEL_CARD.md`.

## Python and JavaScript dependencies

Additional third-party dependencies are declared in files including:

- `backend/pyproject.toml`
- `backend/mlsimkit_dist/requirements_mlsimkit.txt`
- `backend/mlsimkit_dist/pip_freeze_full.txt`
- `frontend/package.json`
- `infra/cdk/requirements.txt`

Each dependency remains subject to its own license. Package metadata and upstream license files should be retained when redistributing bundled dependencies.

This notice is not a substitute for a complete automated dependency and license inventory. Review the dependency manifests and generated deployment artifacts before each public release.
