# FewShotLearning

Project contains implemention of a Few-Shot learning process

1. Install dependencies

Required **python** version >= 3.11 <br>
Replace <cuda_version> with your version
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu<cuda_version>
```

2. Running training
```
python3 train.py [parameters]
```

|Name|Required|Default|Description|
|:---|:------:|:-----:|:-----------|
|dataset_path|True| - |Path to the training dataset|
|patience|False|5|Number of epoch to wait for model improvement|
|save_path|False|./models|Directory where the model will be saved|

3. Running evaluation
```
python3 eval.py [parameters]
```
|Name|Required|Default|Description|
|:---|:------:|:-----:|:-----------|
|dataset_path|True| - |Path to the test dataset|
|model_path|True|-|Path to the trained model|
|results_dir|False|./results|Directory path where the results of a model will be saved|