# 基于自定义 Word2Vec + BiLSTM 的 20 Newsgroups 二分类项目

## 项目简介

这是一个基于 **fetch_20newsgroups** 子数据集完成的文本分类项目，任务是对两类新闻文本进行二分类。项目整体采用了如下技术路线：

- 使用清洗后的训练语料训练 **自定义 Word2Vec** 词向量
- 将训练好的词向量初始化到 **Embedding 层**
- 使用 **Bidirectional LSTM（双向 LSTM）** 建模上下文语义信息
- 在测试集上完成分类评估

本项目最终在测试集上取得了 **77.41% 的准确率**，达到了并超过了 **75%** 的目标要求。

---

## 项目亮点

- 不是直接使用 `nn.Embedding` 随机初始化，而是先训练 **自己的 Word2Vec 词向量**
- 使用 **预训练词向量 + BiLSTM** 的组合方式进行文本分类
- 对空样本、超长文本、PAD/UNK 向量等细节都进行了处理
- 在测试集上取得了 **77%+** 的准确率，具有较好的实验完整性

---

## 数据集说明

本项目使用的数据来自 `fetch_20newsgroups`，并抽取其中两个类别构成二分类任务。根据你的项目描述，该任务是：

- **无神论（atheism）**
- **基督教（christian）**

因此，这是一个典型的英文文本二分类任务。

数据以保存后的形式读取：

```python
from news_data import load_saved_data
import numpy as np

data = load_saved_data("news_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
word_to_idx = data["word_to_idx"]
vocab_size = data["vocab_size"]
```

其中：

- `X_train` / `X_test`：训练集与测试集文本
- `y_train` / `y_test`：对应标签
- `word_to_idx`：词表到索引的映射
- `vocab_size`：词表大小

---

## 整体流程

```text
原始文本
   ↓
删除空样本
   ↓
使用训练集文本训练自定义 Word2Vec
   ↓
构造 embedding_matrix（对齐分类任务词表）
   ↓
文本转索引
   ↓
按统一长度 padding / truncation
   ↓
构建 TensorDataset 和 DataLoader
   ↓
BiLSTM 训练
   ↓
测试集评估
   ↓
保存模型
```

---

## 1. 数据预处理

### 1.1 删除空样本

首先对训练集进行空样本清理。代码中对两种情况做了判断：

1. 文本是字符串且去掉空格后为空
2. 文本已经是序列形式，并且长度为 0 或全部为 0

核心逻辑如下：

```python
new_X_train = []
new_y_train = []

for x, y in zip(X_train, y_train):
    is_empty = False

    if isinstance(x, str):
        if not x.strip():
            is_empty = True
    else:
        arr = np.array(x)
        if len(arr) == 0 or np.all(arr == 0):
            is_empty = True

    if not is_empty:
        new_X_train.append(x)
        new_y_train.append(y)
```

处理结果：

- 原始训练样本数：**1079**
- 删除后训练样本数：**1058**
- 删除的空样本数：**21**

这一步非常重要，因为空样本会影响后续的分词、索引转换、padding 以及 LSTM 长度计算。

---

### 1.2 构建 Word2Vec 训练语料

在删除空样本后，使用训练集文本构建 Word2Vec 的训练语料：

```python
PAD_IDX = word_to_idx["<PAD>"]
UNK_IDX = word_to_idx["<UNK>"]

train_corpus = [text.split() for text in X_train if text.strip() != ""]
```

这里做法很直接：

- 每条文本按空格切分为 token 序列
- 仅使用训练集构建语料，避免测试集信息泄漏

语料统计结果：

- Word2Vec 训练句子数：**1058**

---

### 1.3 训练自定义 Word2Vec

项目没有直接调用外部预训练词向量，而是使用当前训练语料训练自己的 Word2Vec：

```python
from gensim.models import Word2Vec

embed_dim = 256

w2v_model = Word2Vec(
    sentences=train_corpus,
    vector_size=embed_dim,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=10
)
```

参数说明：

- `vector_size=256`：词向量维度为 256
- `window=5`：上下文窗口大小为 5
- `min_count=1`：保留所有出现过的词
- `sg=1`：使用 **Skip-gram**
- `epochs=10`：训练 10 轮

训练结果：

- Word2Vec 词表大小：**15822**

这里的 Word2Vec 词表大小和后续分类模型的 `word_to_idx` 大小不完全一致是正常的：

- Word2Vec 的词表来自当前清洗后的训练语料
- 分类模型的词表来自项目预处理阶段保存的 `word_to_idx`
- 最终需要做的是：**把 Word2Vec 向量对齐到分类模型使用的词表**

---

### 1.4 构造 embedding matrix

为了让 BiLSTM 的 Embedding 层能够直接加载 Word2Vec 结果，需要构造一个与 `word_to_idx` 对齐的矩阵：

```python
embedding_matrix = np.random.normal(
    loc=0.0,
    scale=0.1,
    size=(len(word_to_idx), embed_dim)
).astype(np.float32)

embedding_matrix[PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)

valid_vectors = []
for word, idx in word_to_idx.items():
    if word in ["<PAD>", "<UNK>"]:
        continue
    if word in w2v_model.wv:
        valid_vectors.append(w2v_model.wv[word])

if len(valid_vectors) > 0:
    embedding_matrix[UNK_IDX] = np.mean(valid_vectors, axis=0)
```

处理策略如下：

- 初始矩阵使用正态分布随机初始化
- `<PAD>` 的向量强制设为全 0
- `<UNK>` 的向量用已有有效词向量的平均值填充
- 对于 `word_to_idx` 中存在、但 Word2Vec 中没有的词，保留随机初始化结果

最终结果：

- `embedding_matrix.shape = (11753, 256)`

这说明分类任务最终使用的词表大小为 **11753**，每个词对应 **256 维** 向量。

---

### 1.5 文本转索引

在送入神经网络之前，需要将原始文本转换为整数索引序列：

```python
def texts_to_indices(texts, word_to_idx, unk_idx):
    sequences = []
    for text in texts:
        tokens = text.split()
        seq = [word_to_idx.get(token, unk_idx) for token in tokens]
        sequences.append(seq)
    return sequences
```

然后执行：

```python
X_train_idx = texts_to_indices(X_train, word_to_idx, UNK_IDX)
X_test_idx = texts_to_indices(X_test, word_to_idx, UNK_IDX)
```

这样就把文本转换成了适合 PyTorch Embedding 层输入的索引序列。

---

### 1.6 统计长度并确定 max_len

文本长度差异非常大，因此不能简单使用训练集最大长度，否则会带来严重的显存和计算浪费。

项目先统计了序列长度：

```python
train_lengths = [len(seq) for seq in X_train]
test_lengths = [len(seq) for seq in X_test]
```

统计结果：

- 训练集样本数：**1058**
- 测试集样本数：**717**
- 训练集最长长度：**45731**
- 测试集最长长度：**28696**
- 训练集平均长度：**1283.70**

可以看到文本长度分布极不均衡，因此项目没有直接取最大长度，而是采用了更合理的策略：

```python
max_len = int(np.percentile(train_lengths, 99.5))
print("max_len =", max_len)
```

最终得到：

- `max_len = 12687`

这种做法的优点是：

- 保留绝大多数样本信息
- 避免极端超长文本带来过多 padding 或过高计算成本
- 在效果与效率之间取得平衡

---

### 1.7 Padding / Truncation

定义 padding 函数，将所有样本处理成相同长度：

```python
def pad_sequences(sequences, max_len, pad_value=0):
    padded_sequences = []

    for seq in sequences:
        seq = list(seq)

        if len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        padded_sequences.append(seq)

    return np.array(padded_sequences, dtype=np.int64)
```

执行后：

```python
X_train_pad = pad_sequences(X_train_idx, max_len=max_len, pad_value=PAD_IDX)
X_test_pad = pad_sequences(X_test_idx, max_len=max_len, pad_value=PAD_IDX)
```

结果：

- `X_train_pad.shape = (1058, 12687)`
- `X_test_pad.shape = (717, 12687)`

标签也转换为 NumPy 数组：

```python
y_train = np.array(y_train)
y_test = np.array(y_test)
```

---

### 1.8 转为 Tensor，并过滤零长度样本

之后将数据转换为 PyTorch Tensor：

```python
X_train_tensor = torch.tensor(X_train_pad, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
```

并通过 `PAD_IDX` 统计每条样本的真实长度：

```python
train_lengths = (X_train_tensor != PAD_IDX).sum(dim=1)
test_lengths = (X_test_tensor != PAD_IDX).sum(dim=1)
```

然后进一步过滤长度为 0 的样本：

```python
train_mask = train_lengths > 0
test_mask = test_lengths > 0
```

这一步的意义是：

- 避免空序列进入 `pack_padded_sequence`
- 保证 LSTM 输入合法
- 提高训练和评估稳定性

从最终分类报告的 `support=695` 可以看出，最终参与测试评估的有效测试样本数为 **695**。

---

## 2. 模型设计：Bidirectional LSTM

本项目的分类模型是一个 **单层双向 LSTM**。核心思想如下：

1. 文本索引先通过 Embedding 层映射为词向量
2. 使用双向 LSTM 同时建模正向和反向语义信息
3. 取最后时刻的正向隐藏状态 `h_forward` 与反向隐藏状态 `h_backward`
4. 拼接后送入全连接层完成二分类

模型实现如下：

```python
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        pad_idx,
        num_classes=2,
        pretrained_embeddings=None,
        freeze_embedding=False
    ):
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.tensor(pretrained_embeddings, dtype=torch.float32),
                freeze=freeze_embedding,
                padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=pad_idx
            )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.init_forget_gate_bias(value=1.0)
```

### 模型结构说明

#### （1）Embedding 层

这里使用了：

```python
nn.Embedding.from_pretrained(...)
```

表示 Embedding 层由自定义 Word2Vec 词向量初始化。

并且设置：

```python
freeze_embedding=False
```

这意味着：

- 初始值来自 Word2Vec
- 训练过程中 Embedding 参数仍然可以继续微调

这通常比“完全冻结词向量”更适合分类任务。

#### （2）双向 LSTM

```python
self.lstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)
```

由于 `bidirectional=True`，所以每个时间步都有：

- 正向隐状态
- 反向隐状态

最后输出的句向量维度是 `hidden_dim * 2`。

#### （3）遗忘门偏置初始化

项目还手动初始化了 forget gate bias：

```python
def init_forget_gate_bias(self, value=1.0):
    hidden_dim = self.lstm.hidden_size
    for name, param in self.lstm.named_parameters():
        if "bias" in name:
            param.data[hidden_dim:2 * hidden_dim].fill_(value)
```

这是一种比较常见的 LSTM 初始化技巧，可以帮助模型在训练早期更稳定地保留信息。

---

## 3. 前向传播设计

为了让 LSTM 只处理真实长度部分，而不是浪费在大量 padding 上，模型前向传播里使用了：

```python
packed = nn.utils.rnn.pack_padded_sequence(
    x_embed, lengths.cpu(), batch_first=True, enforce_sorted=False
)
```

然后取双向 LSTM 的最终隐藏状态：

```python
_, (h_n, c_n) = self.lstm(packed)

h_forward = h_n[0]
h_backward = h_n[1]

h_concat = torch.cat([h_forward, h_backward], dim=1)
logits = self.fc(h_concat)
```

这样做的优点是：

- 避免 padding 位置影响序列建模
- 充分利用双向上下文信息
- 对长文本分类任务更友好

---

## 4. 数据加载与训练设置

项目使用 `TensorDataset + DataLoader` 构建训练和测试迭代器：

```python
train_dataset = TensorDataset(X_train_tensor, train_lengths, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, test_lengths, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

训练配置如下：

- `embed_dim = 256`
- `hidden_dim = 256`
- `num_classes = 2`
- `batch_size = 64`
- `epochs = 20`
- `optimizer = Adam`
- `learning_rate = 1e-3`
- `loss function = CrossEntropyLoss`
- `freeze_embedding = False`

模型实例化代码：

```python
model = BiLSTMClassifier(
    vocab_size=len(word_to_idx),
    embed_dim=embed_dim,
    hidden_dim=256,
    pad_idx=PAD_IDX,
    num_classes=2,
    pretrained_embeddings=embedding_matrix,
    freeze_embedding=False
).to(device)
```

---

## 5. 训练过程

训练阶段每个 epoch 都会统计：

- 训练损失 `train_loss`
- 训练准确率 `train_acc`
- 测试损失 `test_loss`
- 测试准确率 `test_acc`

从日志来看，模型训练非常充分，训练集准确率最终接近 **100%**，测试集准确率稳定在 **0.77 左右**。

部分训练日志如下：

```text
Epoch [1/20]  Train Loss: 0.6939, Train Acc: 0.5350 | Test Loss: 0.6695, Test Acc: 0.5971
Epoch [2/20]  Train Loss: 0.6578, Train Acc: 0.6144 | Test Loss: 0.6686, Test Acc: 0.6345
Epoch [3/20]  Train Loss: 0.6241, Train Acc: 0.7004 | Test Loss: 0.6401, Test Acc: 0.6173
Epoch [4/20]  Train Loss: 0.4922, Train Acc: 0.7722 | Test Loss: 0.6177, Test Acc: 0.6676
Epoch [5/20]  Train Loss: 0.3108, Train Acc: 0.8941 | Test Loss: 0.9806, Test Acc: 0.6647
Epoch [6/20]  Train Loss: 0.2539, Train Acc: 0.9253 | Test Loss: 0.7293, Test Acc: 0.6590
Epoch [7/20]  Train Loss: 0.3137, Train Acc: 0.8658 | Test Loss: 0.5572, Test Acc: 0.7223
Epoch [8/20]  Train Loss: 0.1268, Train Acc: 0.9707 | Test Loss: 0.5668, Test Acc: 0.7856
...
Epoch [20/20] Train Loss: 0.0011, Train Acc: 1.0000 | Test Loss: 0.9252, Test Acc: 0.7741
```

可以观察到：

- 第 8 轮测试准确率达到 **78.56%**
- 最终第 20 轮测试准确率为 **77.41%**
- 训练集准确率升到 100%，说明模型已经学得很充分
- 训练集和测试集之间存在一定差距，说明后期有一定过拟合现象

这也是文本分类任务中较常见的情况。

---

## 6. 测试集结果分析

最终评估结果：

```text
Test Loss: 0.9252303676171736
Test Accuracy: 0.7741007194244605
```

也就是说：

- **测试集损失：0.9252**
- **测试集准确率：77.41%**

这已经明确说明：

> 本项目测试集准确率超过 75%，最终达到约 77%。

分类报告如下：

```text
              precision    recall  f1-score   support

           0     0.7584    0.7267    0.7422       311
           1     0.7859    0.8125    0.7990       384

    accuracy                         0.7741       695
   macro avg     0.7721    0.7696    0.7706       695
weighted avg     0.7736    0.7741    0.7736       695
```

### 结果解读

#### 类别 0

- Precision = **0.7584**
- Recall = **0.7267**
- F1-score = **0.7422**

说明模型对类别 0 的识别能力不错，但召回率略低，表示有一部分该类样本被错分到了另一类。

#### 类别 1

- Precision = **0.7859**
- Recall = **0.8125**
- F1-score = **0.7990**

说明模型对类别 1 的表现更好，尤其是召回率较高，能够识别出更多真实的该类样本。

#### 宏平均与加权平均

- Macro F1 = **0.7706**
- Weighted F1 = **0.7736**

整体来看，两类的性能比较接近，没有出现某一类几乎失效的情况，说明模型具备较好的二分类能力。

---

## 7. 模型保存

项目最后使用 `torch.save` 保存了训练好的检查点：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "word_to_idx": word_to_idx,
    "vocab_size": len(word_to_idx),
    "embed_dim": 256,
    "hidden_dim": 256,
    "pad_idx": PAD_IDX,
    "num_classes": 2,
    "max_len": X_train_pad.shape[1],
    "test_acc": test_acc
}

torch.save(checkpoint, "bilstm_text_classifier.pth")
```

建议保存时让 `embed_dim` 和 `hidden_dim` 与实际训练配置保持一致。

> 你的训练代码中模型实际使用的是 `embed_dim=256`、`hidden_dim=256`，因此 README 中建议按这个真实配置进行记录。

---

## 8. 为什么这个方案有效？

本项目效果能够达到 77%+，核心原因主要有以下几点：

### （1）使用自定义 Word2Vec 初始化 Embedding

相比完全随机初始化：

- 词向量在进入分类模型前已经具有一定语义结构
- 可以帮助模型更快收敛
- 对小规模训练集更友好

### （2）BiLSTM 能同时利用前后文

普通单向 LSTM 只能从左到右建模，而双向 LSTM：

- 能同时使用前文和后文信息
- 更适合捕捉文本分类中的上下文依赖

### （3）使用长度信息和 `pack_padded_sequence`

对长文本任务而言：

- 仅仅 padding 不够
- 还需要显式告诉 LSTM 每个样本的真实长度

这样能够减少无效计算，并降低 padding 对模型的干扰。

### （4）对特殊 token 做了合理处理

- `<PAD>` 置零，避免干扰
- `<UNK>` 使用平均向量，而不是纯随机值
- 对零长度样本进行过滤，保证训练稳定性

---

## 9. 项目可改进方向

虽然当前测试准确率已经达到 **77.41%**，但还有进一步优化空间：

### 1）加入 Dropout

当前模型结构相对简单，后期训练集准确率接近 100%，测试集提升有限，说明存在一定过拟合。可以在以下位置加入 Dropout：

- Embedding 后
- LSTM 输出后
- 全连接层前

### 2）保存最佳模型而不是最后一轮模型

从日志看，第 8 轮测试准确率达到 **78.56%**，高于最终第 20 轮的 **77.41%**。

因此更推荐：

- 每轮评估后记录最佳 `test_acc`
- 只保存测试集表现最好的模型参数

### 3）进一步优化文本长度截断策略

当前使用的是 `99.5%` 分位数截断，已经是比较合理的方案，但仍可以尝试：

- 更小的分位数（如 99%）
- 只保留前半段或前后拼接
- 分层截断策略

### 4）尝试更强的预训练表示

在当前 Word2Vec + BiLSTM 的基础上，还可以继续尝试：

- GloVe
- FastText
- BERT / RoBERTa 等预训练语言模型

---

## 10. 运行环境建议

可以在 `requirements.txt` 中包含如下依赖：

```txt
numpy
scikit-learn
torch
gensim
```

安装示例：

```bash
pip install numpy scikit-learn torch gensim
```

---

## 11. 复现实验的基本步骤

### Step 1：加载保存的数据

```python
data = load_saved_data("news_data.pkl")
```

### Step 2：删除空训练样本

```python
X_train = new_X_train
y_train = new_y_train
```

### Step 3：基于训练文本训练 Word2Vec

```python
w2v_model = Word2Vec(...)
```

### Step 4：构造 `embedding_matrix`

```python
embedding_matrix = ...
```

### Step 5：文本转索引并 padding

```python
X_train_idx = texts_to_indices(...)
X_test_idx = texts_to_indices(...)
X_train_pad = pad_sequences(...)
X_test_pad = pad_sequences(...)
```

### Step 6：构建 DataLoader

```python
train_loader = DataLoader(...)
test_loader = DataLoader(...)
```

### Step 7：训练 BiLSTM

```python
for epoch in range(epochs):
    ...
```

### Step 8：在测试集评估并输出分类报告

```python
print(classification_report(...))
```

### Step 9：保存模型

```python
torch.save(checkpoint, save_path)
```

---

## 12. 总结

本项目完整实现了一个 **从文本预处理、Word2Vec 训练、Embedding 初始化、BiLSTM 建模，到测试评估与模型保存** 的文本分类流程。

最终结果如下：

- 使用 **自定义 Word2Vec** 初始化词向量
- 使用 **Bidirectional LSTM** 进行二分类
- 测试集最终准确率达到 **77.41%**
- 已经超过 **75%** 的预期目标

如果把这个项目放到 GitHub 上，它不仅能体现你会使用深度学习做文本分类，也能体现你对完整实验流程的理解，包括：

- 数据清洗
- 词向量训练
- 序列建模
- 变长文本处理
- 模型评估
- 模型保存

这是一个结构完整、实验结果明确、适合作为课程项目或个人作品集展示的 NLP 项目。

---

## 13. 可直接放在仓库首页的简短介绍

你也可以把下面这段放在 README 最前面作为摘要：

> This project implements a binary text classification pipeline on a subset of the 20 Newsgroups dataset using custom Word2Vec embeddings and a Bidirectional LSTM. After cleaning the data, training Word2Vec on the training corpus, aligning embeddings to the task vocabulary, and training a BiLSTM classifier, the model achieves **77.41% test accuracy**, exceeding the 75% target.

