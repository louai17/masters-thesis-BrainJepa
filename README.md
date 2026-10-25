# Brain Representation Alignment: Evaluating Visual Models Against Human Brain Activity

Master's thesis project in Computer Science (AI specialization) at Goethe University Frankfurt.

## Overview

This project investigates how well visual representations learned by different machine learning models align with human brain activity. It compares representations from self-supervised image models, vision-language models, and video models using functional MRI (fMRI) data.

<<<<<<< HEAD
Acknowledgement: This project is built upon Net2Brain framework developed by the cvai-roig-lab. The original project is available at https://github.com/cvai-roig-lab/Net2Brain.
=======
The project builds on the [Net2Brain framework](https://github.com/cvai-roig-lab/Net2Brain) and uses the Natural Scenes Dataset (NSD) and associated shared data for brain-alignment evaluation.
>>>>>>> e325ca9 (Add thesis analysis notebooks and project README)

## Research Objectives

- Compare brain alignment across different model architectures and learning objectives.
- Investigate representations from image, vision-language, and video models.
- Evaluate the relationship between model representations and human brain activity across brain regions and subjects.
- Examine which models and representations best capture aspects of human visual processing.

## Models

The experiments cover several families of visual representation models.

**Self-supervised image models**
- LeJEPA
- MAE (Masked Autoencoders)
- iBOT
- DINO

**Vision-language models**
- CLIP
- OpenCLIP
- InfoNCE-based representations
- LeVLJEPA

**Video models**
- Video MAE
- DINOv2
- V-JEPA
- LeVJepa

Additional baseline models and model variants are included where applicable. See the individual notebooks for the configurations and experiments they cover.

## Methodology

The evaluation pipeline includes the following methods:

- **Feature extraction:** Extracting intermediate or final representations from visual models.
- **Representational Similarity Analysis (RSA):** Comparing representational dissimilarity structures derived from model features and brain activity.
- **Linear encoding:** Evaluating how well model features predict measured brain responses using linear models.
- **Unique Variance Partitioning (UVP):** Examining the unique contributions of model representations to explained variance.
- **Image-wise ranking:** Analyzing images according to their encoding results to investigate differences in prediction performance.

The experiments use brain regions of interest (ROIs) and data from multiple participants where applicable.

## Repository Structure

| Notebook | Description |
|---|---|
| `3_My_Evaluation.ipynb` | Main evaluation workflow |
| `4_MAE_DINO_IBOT.ipynb` | Comparison of image representation models |
| `5_VL Models.ipynb` | Vision-language model experiments |
| `6_video models.ipynb` | Video representation model experiments |
| `7_JEPA.ipynb` | JEPA-related experiments |
| `8_UVP.ipynb` | Unique variance partitioning analyses |
| `9_Img_Ranking.ipynb` | Image-wise ranking analysis |

## Framework and Data

This project uses [Net2Brain](https://github.com/cvai-roig-lab/Net2Brain) for parts of the feature extraction and brain-alignment evaluation workflow.

The experiments use the [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/) and associated shared data. Consult the dataset's terms and access conditions before obtaining or using the data.

The dataset, pretrained model weights, and other large external assets are not included in this repository.

## Reproducibility

The notebooks document the experimental workflows and analyses. Reproducing the experiments may require:

- Python and the appropriate project dependencies.
- Access to the relevant datasets.
- The pretrained model weights or model implementations required for each experiment.
- A suitable compute environment, depending on the model and experiment.

Exact dependency versions and execution instructions can be added as the repository is further prepared for reproducibility.

## Acknowledgments

This research builds on the Net2Brain framework developed by its original authors and contributors. Please consult the original repository for its license and citation instructions.

The models and datasets used in this research are credited to their respective authors and providers.

## Disclaimer

This repository contains code and experimental analyses developed or assembled as part of a Master's thesis. It is intended for research and educational purposes.

The presence of an experiment in this repository does not imply that all experiments are directly comparable or that their results establish a universal ranking of models.