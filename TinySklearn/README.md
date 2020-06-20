# tinysklearn
仿制了部分sklearn接口，结构简单，容易理解，非常适合机器学习入门
- [项目地址](https://github.com/WhatAboutMyStar/MachineLearningAlgorithm)
## 使用方法
- 安装
```
pip install tinysklearn
```
- 使用 <br>
使用方法和sklearn是一致的(至少实现了fit,predict,transform,score这四种方法)
```
from tinysklearn.tinysklearn import LinearRegression
from tinysklearn.datasets import load_boston
from tinysklearn.preprocessing import StandardScaler
from tinysklearn.neighbors import KNeighborsClassifier
from tinysklearn.model_selection import train_test_split
from tinysklearn.decomposition import PCA
from tinysklearn.metrics import mean_absolute_error

#读取数据
boston = load_boston()
x = boston.data
y = boston.target

#分割训练集测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666666)

#构建模型,训练
lr = LinearRegression()
lr.fit(x_train, x_test)

#预测
lr.predict(x_test)

#评估
lr.score(x_test, y_test)
```

