# Two-Stream-Recurrent-Neural-Networks
Reimplement '*Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks*' using PyTorch framework

## Method
*Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks*, Hongsong Wang, Liang Wang
- [2017 CVPR paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Modeling_Temporal_Dynamics_CVPR_2017_paper.pdf)
- [Basic implementation of RNN on Github](https://github.com/hongsong-wang/RNN-for-skeletons) ( based on Lasagne )

## Dataset
NTU RGB+D Dataset
- [project page](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)
- [github reference](https://github.com/shahroudy/NTURGB-D)
- Only skeleton data is needed for this implementation.

  Download skeleton data and then put them in 'nturgb+d_skeletons' file.

## Usage
### Our Training Platform:  HUAWEI Cloud
Because the computation is too large for our laptop, we use the GPU of HUAWEI Cloud for training.
1. Follow the link below to perform human skeleton data preprocessing.
  - [Link](https://github.com/wyy27)
  - Now we get 'all_train_sample.pkl' and 'all_test_sample.pkl' files.

2. Create a Bucket named *two-stream-rnn*
   - Enter *Object Storage Service* in the [HUAWEI CLOUD Console](https://console.huaweicloud.com/console)
   - Click the *Create Bucket*

3. Upload Object to our bucket
   - Upload all the code and dataset files to *two-stream-rnn* bucket
   - The structure of *two-stream-rnn* bucket is like [Data Tree](https://github.com/wyy27/Two-Stream-Recurrent-Neural-Networks/blob/main/README.md#data-tree) .

4. Create a Training Job
   - Enter *ModelArts* in the HUAWEI CLOUD Console
   - Choose the '*Training Management*' - '*Training Jobs*', click *Create*
   - Configure:
   
          - AI Engine: PyTorch | PyTorch-1.3.0-python3.6
          - Code Directory: /two-stream-rnn/code/
          - Boot File: /two-stream-rnn/code/main.py
          - Training Dataset: /two-stream-rnn/data/
          - Training Output Path: /two-stream-rnn/output/
          - Log Output Path: /two-stream-rnn/log/

### Start Training
In the *two-stream-rnn* bucket:
- We can get the best model in *output* file
- Get Logs in *log* file

## Data Tree
```bash
├── Two_Stream_Recurrent_Neural_Networks
    ├── code
    |   ├── main.py
    |   ├── model.py
    |   └── ntu_rgb_preprocess.py
    ├── data
    |   ├── README.md
    |   ├── nturgb+d_skeletons
    |   |         ├── S001C001P001R001A001.skeleton
    |   |         ├──...
    |   |         └──...
    |   ├── ntu_dataset_main.py
    |   ├── all_train_sample.pkl
    |   └── all_test_sample.pkl
    ├── output
    └── log
```
- Executing ntu_dataset_main.py will generate 'all_train_sample.pkl' and 'all_test_sample.pkl' files.

## Result
| Datasets | Model | Accuracy (Our Model, Paper Model) | Parameters
| :---: | :---: | :---: | :---: |
NTU RGB+D | Temporal RNN (Hierarchical RNN) | 59.81%, 67.8% | 313 M
NTU RGB+D | Spatial RNN (Traversal Sequence)| 61.03%, 55.2% | 229 M
NTU RGB+D | Two-Stream RNN, 3D transform (x) | 68.32%, 68.6% | 542 M
NTU RGB+D | Two-Stream RNN, 3D transform (o) | - , 71.3% | -



