# Reconstruction of Neutrino Momentum and $W$-Boson Kinematics in $pp\rightarrow WH\rightarrow(\ell\nu)(b\bar{b})$ Events Using Generative Machine Learning

This repository consists of two branches:
- master: codes for normalising flow
- diffusion: codes for diffusion models

Training and testing data available on [salsatosi/SHP_neutrino_reconstruct_ppWH](https://huggingface.co/datasets/salsatosi/SHP_neutrino_reconstruct_ppWH) on HuggingFace. Save these datasets in your .root_folder/data.

## Instruction
Please ensure the training and testing datasets are saved in .root_folder/data.
### 1. Model Training (train_nflow.py or train_diffusion.py)
The hyperparameters are set by config_nflow.json or config_diffusion.json.  In the training code, change the number of features by commenting and uncommenting corresponding codes including FEATURES. The output models are saved in .root_folder/models, and the scalers are saved in .root_folder/scalers.

### 2. Model Evaluation
Most of the statistical tests and physics reconstruction tests are performed on diagnostics.py (filename is different for nflows and diffusion). These includes:
- Coverage test
- Event posterior plot
- Correlation test
- Invariant mass reconstruction
- Angular observable comparison

To run a specific test in diagnostics.py, you may want to uncomment the corresponding function in the lower main code part, and comment out other irrelevant tests to save running time.

The reconstruction of polarisation distribution is performed on eval_polarization.py. The permutation features importance test is performed on feature_significance_test.py.
