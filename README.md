# Two-Stream-Recurrent-Neural-Networks
Reimplement '*Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks*' using PyTorch framework

## Usage
1. Follow the link below to perform human skeleton data preprocessing.
  - [Link](https://github.com/wyy27)
2. Run the main.py.
  ```python
  python main.py
  ```

## Data Tree
```bash
├── Two_Stream_Recurrent_Neural_Networks
    ├── ntu-dataset
    |   ├── README.md
    |   ├── ntu_dataset_main.py
    |   ├── all_train_sample.pkl
    |   └── all_test_sample.pkl
    ├── main.py
    ├── model.py
    └── ntu_rgb_preprocess.py
```
- Executing ntu_dataset_main.py will generate 'all_train_sample.pkl' and 'all_test_sample.pkl' files.


## Method
*Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks*, Hongsong Wang, Liang Wang
- [2017 CVPR paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Modeling_Temporal_Dynamics_CVPR_2017_paper.pdf)
- [Basic implementation of RNN](https://github.com/hongsong-wang/RNN-for-skeletons) ( based on Lasagne )

## Dataset
NTU RGB+D Dataset
- Only skeleton data is needed for this implementation.
- [project page](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)
- [github reference](https://github.com/shahroudy/NTURGB-D)


