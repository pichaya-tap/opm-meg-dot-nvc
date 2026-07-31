# opm-meg-dot-nvc
Multimodal OPM-MEG & HD-DOT Fusion via mSPoC
 
This repository contains the official implementation of a symmetric fusion pipeline for simultaneously acquired OPM-MEG and high-density DOT data. Utilizing the multimodal Source Power Comodulation (mSPoC) framework, this pipeline integrates variable physical sensitivities to identify a unified, localized neurovascular unit on the cortical surface.

## Abstract
We present a unified anatomical and functional framework that integrates electrophysiological and hemodynamic signals into a shared Schaefer 2018 parcellation space. By co-optimizing spatial and temporal filters, our framework significantly enhances functional contrast, yielding single-trial neurovascular responses with a median enhancement of 5.6 dB over conventional anatomical priors. This approach successfully localizes task-evoked functional hubs using a "pairwise positive" sign convention, providing a biologically plausible and interpretable map of the neurovascular unit.

## Key Features
* **Unified Anatomical Modeling:** Parallel paths for BEM (MEG via MNE-Python) and FEM (DOT via Cedalion/NIRFASTer) integrated into a subject-specific dual head model.
* **mSPoC Decomposition:** Symmetric source separation that accounts for the non-linearity of the neural bandpower operator to extract latent NVC components.
* **Sign Alignment:** Implementation of a sign convention to ensure MEG power drops (ERD) and hemodynamic rises (HbO) are represented as unified positive hubs on cortical maps.

## 📁 Repository Structure

```text
.
├── notebooks/
│   ├── 01_preprocess.ipynb      #  Structural processing pipeline including anatomical headmodel generation, source reconstruction using dSPM (MEG) and image reconstruction (DOT).
│   └── 02_mspoc.ipynb           # Multimodal optimization, grid search for optimal time lags (τ), and extraction of spatial patterns (A)
├── utils.py                     # Custom modules for mSPoC, visualization and SNR calculation
├── requirements.txt             # Project package dependencies
└── README.md                    # Project overview
```
## Getting Started
1. Installation
Clone the repository and install the required dependencies (including Cedalion and MNE-Python):
```bash
git clone https://github.com/pichaya-tap/opm-meg-dot-nvc.git

cd opm-meg-dot-nvc

pip install -r requirements.txt
```
2. Usage
The pipeline is designed to be executed sequentially through the provided Jupyter notebooks. T1-weighted MRI data must be processed with Freesurfer and the Harmening/Miklody SPM12-based workflow seperately and organized for integration with the head-modeling scripts.

## Citation
If you use this framework in your research, please cite our forthcoming paper
```bibtex
@article{tappayuthpijarn2026mspoc,
  title={Data-driven multimodal fusion of OPM-MEG and DOT resolves individual neurovascular coupling at the single trial level during motor tasks},
  author={Tappayuthpijarn, Pichaya and Co-authors},
  journal={Imaging Neuroscience},
  year={2026}
}
```

## 🤝 Acknowledgments
This work utilizes the [Cedalion](https://github.com/Cedalion-fNIRS/cedalion) toolbox for fNIRS/DOT analysis and image reconstruction and the [MNE-Python](https://mne.tools/stable/index.html)  for electrophysiological analysis and source reconstruction. Specialized head tissue segmentation was performed using the [MRIsegmentation](https://doi.org/10.5281/zenodo.7357674) pipeline (Harmening & Miklody, 2022).
