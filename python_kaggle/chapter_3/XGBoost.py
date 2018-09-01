import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 分离数据特征与预测目标。
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis = 1)

# 对对缺失数据进行填充。
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)
# 随机采样
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
# 对原数据进行特征向量化处理
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
# 采用默认配置随机森林对测试器进行预测
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, y_test))

# 使用XGB会有以下错误，DeprecationWarning: The truth value of an empty array is ambiguous.
# 参照这个https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
# 本文选择忽略这个信息
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# 采用默认配置XGBoost对测试器进行预测
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))


