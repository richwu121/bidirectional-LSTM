# Bidirectional LSTM for 20 Newsgroups Binary Text Classification

## 项目简介

本项目基于 **20 Newsgroups** 数据集，完成一个二分类文本分类任务，目标是区分以下两个类别：

- `alt.atheism`
- `soc.religion.christian`

模型采用 **Bidirectional LSTM（双向 LSTM, BiLSTM）**，通过词嵌入层将文本表示为向量序列，再使用双向 LSTM 提取上下文信息，最后将正向和反向隐藏状态拼接后送入全连接层完成分类。

这个项目覆盖了一个完整的 NLP 基础流程：

1. 数据加载与清洗
2. 文本转索引
3. 序列 padding
4. BiLSTM 模型搭建
5. 模型训练与测试
6. 分类结果分析
7. 模型参数保存

---

## 数据集说明

项目使用 `sklearn.datasets.fetch_20newsgroups` 加载 20 Newsgroups 数据集，并只保留两个类别进行二分类：

- `alt.atheism`
- `soc.religion.christian`

在数据加载时，移除了：

- `headers`
- `footers`
- `quotes`

这样可以减少邮件头、签名和引用内容对分类结果的干扰，使模型更关注正文语义。

从代码实现来看，数据预处理函数主要完成了以下工作：

- 文本全部转为小写
- 去除 HTML 标签
- 去除标点符号
- 去除数字
- 去除多余空格
- 构建词表，并保留出现频率不少于 2 次的词
- 设置特殊标记：`<PAD>` 和 `<UNK>`

---

## 数据预处理流程

### 1. 文本清洗

`preprocess_text(text)` 的作用是将原始文本标准化，减少噪声。主要步骤如下：

```python
text = text.lower()
text = re.sub(r'<[^>]+>', '', text)
text = text.translate(str.maketrans('', '', string.punctuation))
text = re.sub(r'\d+', '', text)
text = ' '.join(text.split())
```

这样做的目的是：

- 降低大小写差异带来的词表膨胀
- 去除无意义符号
- 让相同词语尽量映射到相同 token

### 2. 构建词表

在 `build_vocab(texts)` 中，程序会统计所有词的频率，并只保留频率大于等于 2 的词。词表中包含两个特殊符号：

- `<PAD>`：索引为 0，用于补齐序列长度
- `<UNK>`：索引为 1，用于表示未登录词

这一设计可以减少低频词带来的稀疏问题。

### 3. 数据规模统计

从当前运行结果可以看到：

- 训练集样本数：`1079`
- 测试集样本数：`717`
- 训练集最长长度：`45731`
- 测试集最长长度：`28696`
- 训练集平均长度：`1258.72`

这说明文本长度差异非常大，存在明显长文本，因此如果直接按照最大长度进行 padding，会带来很大的显存和计算开销。

### 4. 文本转索引

函数 `texts_to_indices(texts, word_to_idx, unk_idx)` 的作用是把分词后的文本映射成整数序列：

```python
seq = [word_to_idx.get(token, unk_idx) for token in tokens]
```

这样就把原始字符串文本转换成了可以送入神经网络的数字序列。

### 5. Padding 与长度截断

由于 LSTM 需要 batch 中样本长度一致，因此定义了 `pad_sequences()`：

- 如果序列长度小于 `max_len`，就在末尾补 `<PAD>`
- 如果序列长度大于 `max_len`，就截断到 `max_len`

当前项目没有直接使用最大长度 `45731`，而是使用：

```python
max_len = int(np.percentile(train_lengths, 99.5))
```

最终得到：

- `max_len = 12507`

这种做法比直接取最大长度更合理，因为它避免了极少数超长文本显著拉高计算成本。

padding 后数据维度为：

- `X_train_pad.shape = (1079, 12507)`
- `X_test_pad.shape = (717, 12507)`

标签维度为：

- `y_train.shape = (1079,)`
- `y_test.shape = (717,)`

---

## 模型结构解析

### 1. 模型整体结构

项目定义了 `BiLSTMClassifier(nn.Module)`，整体结构如下：

```python
Embedding -> Bidirectional LSTM -> Concatenate(h_forward, h_backward) -> Linear -> logits
```

### 2. Embedding 层

```python
self.embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embed_dim,
    padding_idx=pad_idx
)
```

作用：

- 把每个词索引映射为稠密向量
- `padding_idx=pad_idx` 可以保证 `<PAD>` 对应的向量不会参与有效语义学习

当前设置中：

- `embed_dim = 512`

### 3. 双向 LSTM 层

```python
self.lstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)
```

这里使用的是 **单层双向 LSTM**：

- 正向 LSTM 从前到后读取句子
- 反向 LSTM 从后到前读取句子

这样模型可以同时利用前文和后文信息。

当前设置中：

- `hidden_dim = 512`
- `num_layers = 1`
- `bidirectional = True`

### 4. 只使用最后隐藏状态进行分类

在前向传播中，模型没有使用所有时间步输出，而是直接取最后的双向隐藏状态：

```python
_, (h_n, c_n) = self.lstm(packed)
h_forward = h_n[0]
h_backward = h_n[1]
h_concat = torch.cat([h_forward, h_backward], dim=1)
logits = self.fc(h_concat)
```

其含义是：

- `h_forward`：正向 LSTM 的最后隐藏状态
- `h_backward`：反向 LSTM 的最后隐藏状态
- `h_concat`：将两个方向的语义表示拼接起来，形成一个长度为 `2 * hidden_dim` 的句向量
- 最后输入线性层做二分类

全连接层为：

```python
self.fc = nn.Linear(hidden_dim * 2, num_classes)
```

由于是二分类，因此：

- `num_classes = 2`

### 5. 使用 pack_padded_sequence

模型中使用了：

```python
packed = nn.utils.rnn.pack_padded_sequence(
    x_embed, lengths.cpu(), batch_first=True, enforce_sorted=False
)
```

它的作用是：

- 告诉 LSTM 每个样本的真实长度
- 避免模型把 padding 部分也当成有效输入
- 提高训练效率
- 减少 padding 对隐藏状态的干扰

这是变长文本输入到 RNN/LSTM 时非常重要的一步。

### 6. Forget gate bias 初始化

项目中还专门初始化了 forget gate 的 bias：

```python
self.init_forget_gate_bias(value=1.0)
```

这是一种常见技巧，目的是让 LSTM 在训练初期更倾向于保留信息，有时可以帮助模型更稳定地学习长距离依赖。

---

## 训练流程解析

### 1. 数据张量化

在训练前，代码将 padding 后的数据转换为 PyTorch 张量：

```python
X_train_tensor = torch.tensor(X_train_pad, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
```

同时通过：

```python
train_lengths = (X_train_tensor != PAD_IDX).sum(dim=1)
test_lengths = (X_test_tensor != PAD_IDX).sum(dim=1)
```

来计算每个样本的真实长度。

### 2. 训练配置

项目中的关键训练参数如下：

- 词向量维度：`512`
- 隐藏层维度：`512`
- 类别数：`2`
- 优化器：`Adam`
- 学习率：`1e-3`
- 损失函数：`CrossEntropyLoss`
- 训练轮数：`20`

代码如下：

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
```

### 3. 训练函数

`train_one_epoch()` 的核心流程：

1. 模型设置为训练模式 `model.train()`
2. 从 dataloader 中取出 batch
3. 前向传播得到 `logits`
4. 计算损失 `loss`
5. 反向传播 `loss.backward()`
6. 参数更新 `optimizer.step()`
7. 统计整轮 loss 和 accuracy

### 4. 评估函数

`evaluate()` 的主要流程：

- 使用 `model.eval()` 切换到评估模式
- 用 `torch.no_grad()` 关闭梯度计算
- 统计测试集损失与准确率
- 返回预测标签与真实标签，供 `classification_report` 使用

---

## 实验结果

### 1. 训练过程

从训练日志可以看到：

- 第 1 轮：`Train Acc = 0.6493`，`Test Acc = 0.6590`
- 第 2 轮：`Train Acc = 0.9423`，`Test Acc = 0.7108`
- 第 3 轮：`Train Acc = 0.9905`，`Test Acc = 0.7151`
- 第 4 轮开始训练准确率已经接近或达到 `1.0000`
- 最优测试准确率大约出现在第 14 轮：`0.7309`
- 第 20 轮测试准确率为：`0.7237`

最终测试结果：

- `Test Loss = 1.4110`
- `Test Accuracy = 0.7237`

### 2. 分类报告

当前截图中的 `classification_report` 结果如下：

| 类别 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.7409 | 0.5884 | 0.6559 | 311 |
| 1 | 0.7143 | 0.8333 | 0.7692 | 384 |
| Macro Avg | 0.7276 | 0.7109 | 0.7126 | 695 |
| Weighted Avg | 0.7262 | 0.7237 | 0.7185 | 695 |

如果按照类别顺序理解，通常可以认为：

- `0 -> alt.atheism`
- `1 -> soc.religion.christian`

则可以看出：

- 模型对类别 `1` 的召回率更高，说明它更容易识别 `soc.religion.christian`
- 类别 `0` 的召回率相对较低，说明部分无神论文本被误分类到了另一类

---

## 结果分析

### 1. 模型已经明显过拟合

最明显的现象是：

- 训练准确率非常快地达到 `1.0000`
- 测试准确率却只停留在 `0.72 ~ 0.73`
- 测试损失随着训练轮数增加持续上升

这说明模型在训练集上记忆得非常充分，但泛化能力不足。

### 2. 可能原因

#### （1）模型容量较大

当前模型参数比较大：

- `embed_dim = 512`
- `hidden_dim = 512`

对于 1000 级别样本数的训练集来说，这个规模偏大，容易过拟合。

#### （2）序列长度过长

虽然已经没有使用最大长度 `45731`，但 `12507` 仍然非常长。

这会带来两个问题：

- 模型训练成本高
- 长文本中噪声信息较多，不一定有利于分类

#### （3）当前词表构建方式存在信息泄漏风险

当前代码中使用：

```python
word_to_idx = build_vocab(X_train + X_test)
```

这意味着词表同时使用了训练集和测试集文本构建。这样做虽然方便，但从严格实验角度看，测试集信息提前参与了建模过程，存在轻微数据泄漏风险。

更严谨的做法应该是：

```python
word_to_idx = build_vocab(X_train)
```

然后测试集中的未登录词统一映射到 `<UNK>`。

#### （4）测试样本统计需要再次核对

前面长度统计与 padding 结果显示测试集样本数为 `717`，但当前截图中的分类报告 `support` 合计为 `695`。这说明实际评估时可能存在：

- dataloader 没有覆盖全部测试集
- 某些样本在评估阶段被过滤
- 使用的测试集与前面统计的测试集并非完全一致

这一点建议在提交项目前重新检查，确保最终报告和数据规模一致。

---

## 模型保存

训练结束后，项目将模型保存为：

```python
bilstm_text_classifier.pth
```

保存内容包括：

- `model_state_dict`
- `optimizer_state_dict`
- `word_to_idx`
- `vocab_size`
- `embed_dim`
- `hidden_dim`
- `pad_idx`
- `num_classes`
- `max_len`
- `test_acc`

这种 checkpoint 保存方式比较完整，后续不仅可以恢复模型参数，也能恢复推理时必需的词表和超参数。

---

## 项目优点

这个项目的优点包括：

- 完整实现了一个文本分类任务的标准流程
- 使用了双向 LSTM，能够同时利用前向和后向上下文
- 使用 `pack_padded_sequence` 正确处理变长序列
- 使用 forget gate bias 初始化增强训练稳定性
- 保存了完整 checkpoint，便于后续部署与复现

---

## 可以继续改进的方向

### 1. 仅用训练集构建词表

避免测试集信息泄漏，提高实验严谨性。

### 2. 增加正则化

可以尝试：

- `dropout`
- `weight decay`
- `early stopping`

### 3. 缩短最大长度

可以尝试把 `max_len` 改为更小的值，比如：

- 500
- 1000
- 2000

很多新闻文本的判别信息往往集中在前半部分，这样可能会提高训练效率并缓解过拟合。

### 4. 改进句子表示方式

当前只使用了最后的 `h_forward` 和 `h_backward`。后续可以尝试：

- mean pooling
- max pooling
- attention
- BiLSTM + Attention

### 5. 增加更多评价指标

除了 accuracy，还可以进一步分析：

- confusion matrix
- ROC-AUC
- PR curve

---

## 如何运行项目

### 1. 安装依赖

```bash
pip install numpy scikit-learn torch
```

### 2. 运行数据预处理脚本

```bash
python 20_news_data.py
```

### 3. 运行训练代码

如果你的训练代码写在 notebook 或单独脚本中，可以执行对应文件，完成：

- 数据加载
- 文本转索引
- padding
- 模型训练
- 模型评估
- checkpoint 保存

---

## 项目总结

本项目实现了一个基于 **Bidirectional LSTM** 的二分类文本分类器，用于区分 **无神论** 与 **基督教** 两类新闻文本。模型在训练集上很快达到极高精度，但在测试集上的最终准确率约为 **72.37%**，并表现出较明显的过拟合现象。

从实验结果来看，这个项目已经很好地展示了：

- 文本预处理
- 序列建模
- 双向 LSTM 分类
- 训练与评估
- 模型保存与复现

如果后续加入更严格的数据划分方式、更合理的截断长度和更强的正则化策略，模型性能还有进一步提升空间。

---

## 仓库描述（可选）

可以将 GitHub 仓库简介写为：

> Bidirectional LSTM for binary text classification on the 20 Newsgroups dataset (alt.atheism vs. soc.religion.christian).

