import tensorflow as tf
import numpy as np
# 初始化Tensorflow常量，
greeting = tf.constant('Hello Google Tensorflow!')
# 启动一个会话
sess = tf.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出结果并关闭会话
print(result)
sess.close()


# 使用TensorFlow进行一次线性函数的计算
# 声明1 * 2 向量
matrix1 = tf.constant([[3., 3.]])
# 声明2 * 1 向量
matrix2 = tf.constant([[2.],[2.]])
# 声明矩阵相乘
product = tf.matmul(matrix1, matrix2)
# 声明相加
linear = tf.add(product, tf.constant(2.0))
# 最后计算linear模块的值
with tf.Session() as sess:
    result = sess.run(linear)
    print(result)


# 使用Tensorflow自定义线性分类器用于肿瘤预测
import tensorflow as tf
import numpy as np
import pandas as pd

train = pd.read_csv('../chapter_1/dataset/breast-cancer-train.csv')
test = pd.read_csv('../chapter_1/dataset/breast-cancer-test.csv')

# 分割特征和分类目标
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)
# 定义线性分类器
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, X_train) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_train))
# 使用梯度下降作为优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# 准备测试样本
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 作图
import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s = 200, c = 'red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s = 150, c = 'black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

# 以0.5作为分界
lx = np.arange(0, 12)
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]

plt.plot(lx, ly, color ='green')
plt.show()

from sklearn import datasets, metrics, preprocessing, model_selection

# Load dataset
boston = datasets.load_boston()
X, y = boston.data, boston.target

# Split dataset into train / test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
    test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import skflow
# 这里遇到了很多问题，可能跟使用Tensorflow 1.0之后的关系？
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_predict = tf_lr.predict(X_test)

print('The mean absoluate error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_absolute_error(tf_lr_y_predict, y_test))
print('The mean squared error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_squared_error(tf_lr_y_predict, y_test))
print('The R-squared value of Tensorflow Linear Regressor on boston dataset is', metrics.r2_score(tf_lr_y_predict, y_test))
