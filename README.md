
# UC5 "Deep Image Annotation"
## Project DEEPHEALTH

Repository for the Use Case 5 "Deep Image Annotation" containing:
- models based on the EDDL/ECVL libraries;
- models based on the PyTorch-Lightning library.

Previous models and PyTorch model can be found in the release "V1" available to download.

## FOLDER STRUCTURE

- `conda_envs`: YAML files for creating the environments based on the EDDl/ECVL (cuDNN version) and PyTorch-Lightning
- `demo`, `demo_src`: code to be used in demos
- `src`: source code
    - `src/eddl`: EDDL-based models
    - `src/PyTorch-Lightning`: PyTorch (Lightning) models
    - `src/preproc`: code for pre-processing the IU-CHEST and the MIMIC-CXR datasets

## HOW TO USE
- First, preprocess the datasets. 
    - For the IU-CHEST dataset, use the `Makefile.mk` in `src/preproc`
    - For the MIMIC-CXR dataset, use the Jupyter Notebook `src/preproc/NB_mimic_ds.ipynb`

- Set up the experiments using the Jupyter notebooks `src/eddl/0_experiments.ipynb` and `src/eddl/0_experiments_mimic.ipynb` for, respectively, the IU-CHEST and the MIMIC-CXR datasets. Some experiments are provided in the two notebooks.

- For the experiments already defined in the jupyter notebooks at the item above, use the two makefiles `src/eddl/Makefile.mk` and `src/eddl/Makefile_MIMIC.mk` for, respectively, the IU-CHEST and the MIMIC-CXR datasets.

Once configured an experiment, the correct sequence of the scripts is: `1_train_cnn.py`, `2_train_rnn.py`, `3_test_rnn.py`. 
