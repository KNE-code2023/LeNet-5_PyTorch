# LeNet-5_PyTorch
使用 PyTorch 實現 LeNet-5 神經網路來進行 MNIST 數據集分類

## 訓練模型
```bash
python train.py
```
![Alt text](image0.png)
## 測試模型
```bash
python train.py
```
![Alt text](image1.png)
## 專案結構
* `train.py` ： 用於訓練模型的腳本

* `test.py` ： 用於測試模型的腳本

* `model.py` ： LeNet-5 神經網路的定義

* `data.py` ： 用於下載和處理 MNIST 數據集的腳本

* `config.json` ： 訓練時的超參數設定
## 參考文獻
### **Gradient-based learning applied to document recognition**
_Y. Lecun, L. Bottou, Y. Bengio, P. Haffner_

**Abstract**  

Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day.  
[[Paper]](https://ieeexplore.ieee.org/document/726791)
```bash
@article{LeNet-5,
    title = {Gradient-based learning applied to document recognition},
    author = {Y. Lecun, L. Bottou, Y. Bengio, P. Haffner},
    journal = {IEEE},
    year = {1998}
}
```