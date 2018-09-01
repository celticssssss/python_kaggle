# 导入20类新闻文本进行词向量训练
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target

from bs4 import BeautifulSoup
import nltk, re
# 定义一个函数名为news_to_sentences将每条新闻中的句子逐一剥离出来，并返回句子列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news, "lxml").get_text()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)

    sentences = []

    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())

    return sentences

sentences = []

for x in X:
    sentences += news_to_sentences(x)
print(len(sentences))



# 对不同参数设定值
num_features = 300    # 词向量维度
min_word_count = 20   # 频度
num_workers = 2     # 设定并行训练使用的CPU核心数
context = 5        # 设定上下文的串口大小
downsampling = 1e-3   # 下采样设定

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import word2vec
# 训练词向量模型
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

# 表示当前模型是最终版
model.init_sims(replace=True)
# 寻找与morning最相关的10个词汇
print(model.most_similar('morning'))
# 寻找与email最相关的10个词汇
print(model.most_similar('email'))
