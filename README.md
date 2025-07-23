# UMotion 

Code for our [paper](https://www.arxiv.org/abs/2505.09393) "UMotion: Uncertainty-driven Human Motion Estimation from
Inertial and Ultra-wideband Units"

## Usage

### Install dependencies using uv

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
> check `pyproject.toml` for dependencies.

### Prepare SMPL body model

1. Download [SMPL model](https://smpl.is.tue.mpg.de/download.php). You should download `version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)` since UIP use neutral SMPL model.
2. Put the model file into `models/smpl/`. The directory structure should look like the following:

```
models/smpl/basicModel_f_lbs_10_207_0_v1.1.0.pkl
models/smpl/basicModel_m_lbs_10_207_0_v1.1.0.pkl
models/smpl/basicModel_neutral_lbs_10_207_0_v1.1.0.pkl
```

### Prepare datasets

> Change `base_dir` in `src/config` to your project directory.

Download the dataset and put it into `datasets/raw`. The directory structure should look like the following:

```
datasets/raw/AMASS/<subdata>/...
datasets/raw/DIP_IMU_and_Others/DIP_IMU/s_<subject_id>/...pkl
datasets/raw/TotalCapture_Real_60FPS/...pkl
datasets/raw/uip/[(test)(train)].pt
```

> We provide different process scripts for preprocessing data into the same data structure (run in project root directory).

| Dataset    | Download Link                                                                 | Processing Script                |
|----------------|------------------------------------------------------------------------------|----------------------------------|
| AMASS          | https://amass.is.tue.mpg.de/ (SMPL-H)                                        | `python src/data/amass.py`       |
| DIP-IMU        | https://dip-imu.github.io/ (DIP IMU AND OTHERS - DOWNLOAD SERVER 1)                                        | `python src/data/dipimu.py`      |
| TotalCapture   | https://dip-imu.github.io/ (ORIGINAL TotalCapture DATA W/ CORRESPONDING REFERENCE SMPL Poses)                         | `python src/data/totalcapture.py`|
| UIP            | https://siplab.org/projects/UltraInertialPoser                         | `python src/data/uip.py`         |

### Run the training or evaluation

#### Download pre-trained models

Make sure to have gdown installed

```sh
pip install gdown
```

Then, run the following command line:

```sh
bash download_pretrained_models.sh
```

---

For shape estimator

```sh
python src/main_shape.py
```

For pose estimator (DIP-IMU, TotalCapture)

```sh
python src/main_pose.py # DIP-IMU, TotalCapture, without UKF (distance without noise)
python src/main_uip.py # UIP, with UKF (noisy distance)
python src/main_ukf.py # TotalCapture, with UKF (noisy distance)
```

> Uncomment `do_train()` in `main_shape.py` and `main_pose.py` to train the model.
> The training parameters are set in `configs/config.yaml` and `configs/config_noise.yaml`

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{liu2025umotion,
  title={UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units},
  author={Liu, Huakun and Ota, Hiroki and Wei, Xin and Hirao, Yutaro and Perusquia-Hernandez, Monica and Uchiyama, Hideaki and Kiyokawa, Kiyoshi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7085--7094},
  year={2025}
}
```

## Misc

The code for sensing the distance matrix using the DW3000 is available at [DW3000](https://github.com/kk9six/dw3000.git).

If you encounter any issues or have questions, feel free to open an issue. You may also contact me via the email address: liu.huakun.li0@is.naist.jp.
