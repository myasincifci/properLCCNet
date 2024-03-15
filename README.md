# Dual LCC-Net: Targetless LiDAR-Camera Calibration with Multiple LiDARs

## Description
This repository contains code for our APPRAS project conducted in the WS23 by:

- Bryan Oppong-Boateng
- Daniia Vergazova
- Mehmet Yasin Cifci
- Sönke Nickelsen
- Ziad Abouhalawa

We have implemented an extension of the LCC-Net [1] architecture for usage on a custom real world dataset with two lidar inputs and evaluated it's performance.
The results can be seen in the technical report. 

## Installation
<b>Python 3.7 is required for installation.</b> \
The remaining dependenceies can be installed via: \
    `pip install -r requirements.txt`


## Usage
### Training 
To start training, first make sure to place the dataset into the repositories root directory under `data/kitti_full_rate_025-20240221T190558Z-001`. Once the dataset is present, training can be started with `python train_proper.py`.

### Evaluation
To evaluate the dataset on the validation sequence run `python evaluate.py`, either using your own checkpoints or using our pretrained weights. Either way the checkpoint files need to be present in the `pretrained` directory and in case you are using paths need to be adjusted accordingly in `evaluate.py`.

## Citations
[1] Lv, Xudong, et al. "LCCNet: LiDAR and camera self-calibration using cost volume network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.