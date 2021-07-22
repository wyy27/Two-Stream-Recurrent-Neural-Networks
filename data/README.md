# NTU RGB+D Dataset Preprocessing
- [Dataset website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp)
- [Dataset paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)

## Human Skeleton Data
![25 body joints picture](https://github.com/wyy27/Two-Stream-Recurrent-Neural-Networks/blob/main/data/body_joints.png)

## Usage
- **'Cross subject metric dataset preprocessing' which is the evaluation metric of NTU RGB+D dataset.** <br>
```python
python ntu_dataset_main.py
```
  - Arguments
    - `--file-path` (str): File path where NTU RGB+D data exists
  - We can get two pickle files
    - `all_train_sample.pkl`
    - `all_test_sample.pkl`
