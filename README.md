# CARL-Net

This repository contains the official implementation for the paper titled "CARL-Net".

## 1. Prepare data

Preprocess: refer to the image pre-processing method in [CoraNet](https://github.com/koncle/CoraNet) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) for the Pancreas dataset, Left atrium and ACDC dataset. 

The `dataloaders` folder contains the necessary code to preprocess the Left atrium and ACDC dataset. Pancreas pre-processing code can be obtained from CoraNet.

## 2. Environment

We recommend an environment with Python >= 3.8, and then install the following dependencies:

```bash
pip install -r requirements.txt

## 3. Train / Test

Run the train script on Pancreas-CT dataset.

Train:

bash
python Pancreas_train.py
Test:

bash
python test_Pancreas.py

##4. Acknowledgements
Our code is largely based on SDCL and BCP. Thanks to these authors for their valuable work. We hope our work can also contribute to related research.
