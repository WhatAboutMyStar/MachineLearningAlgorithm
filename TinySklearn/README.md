# tinysklearn
仿制了部分sklearn接口，结构简单，容易理解，非常适合机器学习入门
- [项目地址](https://github.com/WhatAboutMyStar/MachineLearningAlgorithm)
## 使用方法
- 安装
```
pip install tinysklearn
```
- 使用
使用方法和sklearn是一致的(fit,predict,transform,score)
```
from tinysklearn.tinysklearn import LinearRegression
from tinysklearn.datasets import load_boston
from tinysklearn.preprocessing import StandardScaler
from tinysklearn.neighbors import KNeighborsClassifier
from tinysklearn.model_selection import train_test_split
from tinysklearn.decomposition import PCA
from tinysklearn.metrics import mean_absolute_error
```
