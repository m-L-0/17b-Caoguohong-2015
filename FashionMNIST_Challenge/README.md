# FashionMNIST_Challenge 作業說明
### 一. 使用數據集 
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
### 二. 依賴庫
1. tensorflow==1.4.0
2. sklearn==0.0
3. jupyter==1.0.0
4. Cython==0.27.3
5. image==1.5.16
6. matplotlib==2.1.0
7. h5py==2.7.1
8. pydot==1.2.3(Keras保存圖像需要依賴，執行下面👇的命令即可)

```pip install pydot-ng & brew install graphviz```


### 三、参考模型
<a href="http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006" target="_blank">resnet50神經網絡模型圖</a>

### 四、具體文件說明

```
├── K-Means.ipynb   # K-Means 算法文檔
├── KNN.ipynb       # KNN 算法文檔
├── README.md
├── Resnet.ipynb    # 深度殘差網絡文檔說明
├── cnn.py          # 另一個深度學習算法（未運行）
├── img             # 圖片的文件夾
│   ├── 0_Label_4.bmp
│   ├── 1_Label_0.bmp
│   ├── 2_Label_7.bmp
│   ├── 3_Label_9.bmp
│   ├── 4_Label_9.bmp
│   ├── 5_Label_9.bmp
│   ├── Resnet_model.png    # 深度殘差網絡模型圖
│   ├── model.png           # ResNet50 網絡模型
│   ├── model_cnn.png       # 上面的cnn模型
│   ├── srecc.png
├── resnet.h5       # 訓練完成的模型
├── resnet_.h5
├── resnet_xiugai_ken2_1126.h5
├── test.tfrecords  # 測試數據集
├── train.tfrecords # 訓練數據集
└── validation.tfrecords    # 驗證數據集
```
