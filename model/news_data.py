import re
import string
import pickle
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    """简单的文本预处理"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除标点符号
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = ' '.join(text.split())  # 移除多余空格
    return text

def build_vocab(texts):
    """构建词汇表"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)
    
    # 创建词汇表，从1开始（0保留给padding）
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= 2:  # 只保留出现至少2次的词
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx

def load_and_preprocess_data():
    """加载并预处理20newsgroups数据"""
    categories = ['alt.atheism', 'soc.religion.christian']   # 无神论vs基督
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # 预处理文本
    X_train = [preprocess_text(doc) for doc in newsgroups_train.data]
    X_test = [preprocess_text(doc) for doc in newsgroups_test.data]
    
    # 编码标签
    label_encoder = LabelEncoder() # 将文本标签转换为整数
    y_train = label_encoder.fit_transform(newsgroups_train.target)
    y_test = label_encoder.transform(newsgroups_test.target)
    
    # 构建词汇表
    word_to_idx = build_vocab(X_train + X_test)
    vocab_size = len(word_to_idx)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"训练样本数量: {len(y_train)}")
    print(f"测试样本数量: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test, word_to_idx, vocab_size    

def save_data(filepath, X_train, X_test, y_train, y_test, word_to_idx, vocab_size):
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "word_to_idx": word_to_idx,
        "vocab_size": vocab_size,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_saved_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, word_to_idx, vocab_size = load_and_preprocess_data()

    save_data(
        "news_data.pkl",
        X_train, X_test, y_train, y_test, word_to_idx, vocab_size
    )

    print("数据已保存到 20news_data.pkl")