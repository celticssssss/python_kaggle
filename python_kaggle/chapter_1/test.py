import pandas as pd

'''
./表示当前目录
../表示父级目录
/表示根目录

'''
#读取数据，将训练和测试数据导入
df_train = pd.read_csv('./dataset/breast-cancer-train.csv')
df_test = pd.read_csv('./dataset/breast-cancer-test.csv')

#将type作为特征，其余两个作为特征 negative为良性肿瘤，positive为恶性肿瘤
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# print(df_test_negative)

import matplotlib.pyplot as plt
#这个就是画散点图，首先知道横纵坐标，然后标点，重复的不管
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
#横纵注释
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

import numpy as np

#设置两个随机参数
intercept = np.random.random([1])
coef = np.random.random([2])
print(intercept, coef)
#设置x和y的值，因为lx是一个list,所以ly也是一个list
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]
#画图
plt.plot(lx, ly, c='yellow')


plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()


#这里导入sklearn中的逻辑回归器，这里的是一个class
#fit是一个拟合函数，必须要拟合之后才能使用
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# lr.fit(df_train[['Clump Thickness', 'Cell Size']][:2], df_train['Type'][:2])
# print('Testing accuracy (0 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))


#这里是引用class里经过拟合的参数
intercept = lr.intercept_
coef = lr.coef_[0, :]
#本来函数是lx * coef[0] + ly * coef[1] + intercept = 0，转换成ly变成下面的形式
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()


#全部测试
lr_test = LogisticRegression()
# lr.fit(df_train[['Clump Thickness', 'Cell Size']][:2], df_train['Type'][:2])
# print('Testing accuracy (0 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
lr_test.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print('Testing accuracy (all training samples):', lr_test.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr_test.intercept_
coef = lr_test.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()







