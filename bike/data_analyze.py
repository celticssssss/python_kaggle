import pandas as pd
"""
从CSV中读取文件
"""
df_train = pd.read_csv('train.csv',header=0)
"""
将一些分析的数据加入到df_train中
将datetime的信息提出出来并加入到df_train中
"""
df_train['hour']=pd.DatetimeIndex(df_train.datetime).hour
df_train['day']=pd.DatetimeIndex(df_train.datetime).day
df_train['month']=pd.DatetimeIndex(df_train.datetime).month
"""
去掉一些不需要的信息，并且重新排列
"""
df_train.drop(['datetime','casual','registered'],axis=1,inplace=True)
df_train = df_train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','count','month','day','hour']]
"""
将目标信息提取出来
"""
df_train_target = df_train['count'].values
print(df_train_target.shape)
df_train_data = df_train.drop(['count'],axis =1).values
print(df_train_data.shape)

from sklearn import linear_model
from sklearn import model_selection
# 切分一下数据（训练集和测试集）
cv = model_selection.ShuffleSplit( n_splits=3, test_size=0.2,random_state=0)
cv.get_n_splits(df_train_data)
print("岭回归")
for train, test in cv.split(df_train_data):
    svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(svc.score(df_train_data[train], df_train_target[train]),svc.score(df_train_data[test], df_train_target[test])))
