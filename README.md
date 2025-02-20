# RPH-Counter

**[RPH-Counter: Field Detection and Counting of Rice Planthoppers Using a Fully Convolutional Network with Object-Level Supervision](https://pan.quark.cn/s/2399504ff403)**

Zhang Z, Zhan W, Sun K, et al. (2024). RPH-Counter: Field Detection and Counting of Rice Planthoppers Using a Fully Convolutional Network with Object-Level Supervision. Computers and Electronics in Agriculture, 225, 109242.

## Preview

We have open-sourced the test code for RPH-Counter and provided some visualization examples.

Test samples are stored in the `./data` folder. The model can be downloaded from [this link](https://pan.quark.cn/s/2399504ff403). Place the model in the `./checkpoints/2024-02-21_10-40-08` folder. Visualization results will be saved in the `vis` folder.

## Installation

This project has been tested under [PyTorch](https://pytorch.org/) versions 2.1.0 and 2.2.0. First, install [PyTorch](https://pytorch.org/). Then, install the required packages using:

```bash
pip install -r requirements.txt
```

## Quick Run
### Testing & Visualization:

1. Run `test.py`.

    Or run the following command: 
    ```bash
    python test.py --datadir ./data --model fcn_resnet_safpn_s16 --model_path ./checkpoints/2024-02-21_10-40-08/model_best_f1.pth
    ```
    After the run completes, the detection results will be saved in `./checkpoints/2024-02-21_10-40-08/res.json`, and visualization results will be saved in the `vis` folder.


2. Run `./tools/read_my_result_merge_vis.py`. 

    This script will read the detection results from step 1 and restore the split images back to the original images. The visualization results will be saved in the `vis` folder.

## Dataset
The DataLoader loads JSON annotation files labeled by labelme or anylabeling. 
```bash
pip install label
```
```bash
pip install anylabeling
```
If you want to test your own data, refer to the directory structure of `test_data` in the `./data` folder.

## Results
![Results](https://github.com/ZZL0897/RPH-Counter/blob/main/checkpoints/2024-02-21_10-40-08/vis/o_pred/IMG_20230912_105844.jpg)
![Results](https://github.com/ZZL0897/RPH-Counter/blob/main/checkpoints/2024-02-21_10-40-08/vis/o_pred/IMG_20230913_100043.jpg)