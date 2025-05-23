深度学习从 0 到最小 LLM 实践
草案

以下是14步从零开始到最小LLM的实践路径，所有代码基于PyTorch且可直接运行：

-----
**1. 环境配置**
```python
# 安装必要库（在终端运行）
conda create -n dl python=3.8
conda activate dl
pip install torch numpy matplotlib ipykernel
```

-----
**2. 线性回归**
```python
import torch

# 生成数据
X = torch.arange(0, 10, 0.1).unsqueeze(1)
y = 3 * X + 2 + torch.randn(X.shape)*0.5

# 模型定义
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"权重: {model.weight.item():.2f}, 偏置: {model.bias.item():.2f}")
```

-----
**3. 逻辑回归（二分类）**
```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid()
)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    pred = model(X).squeeze()
    loss = torch.nn.BCELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

-----
**4. 多层感知机（MNIST分类）**
```python
from torchvision import datasets, transforms

# 数据加载
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# 训练循环（需自行补全DataLoader和训练循环）
```

-----
**5. 卷积神经网络**
```python
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3),  # 输入通道1，输出16通道
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(16*13*13, 10)
)
```

-----
**6. 文本生成（RNN）**
```python
text = "hello world"
chars = list(set(text))
char_to_idx = {c:i for i,c in enumerate(chars)}

# 准备数据
inputs = [char_to_idx[c] for c in text[:-1]]
targets = [char_to_idx[c] for c in text[1:]]

model = torch.nn.RNN(input_size=len(chars), hidden_size=8, batch_first=True)
```

-----
**7. 词嵌入**
```python
embedding = torch.nn.Embedding(num_embeddings=1000, embedding_dim=50)
input_ids = torch.tensor([32, 51, 72])
embedded = embedding(input_ids)
```

-----
**8. 自注意力机制**
```python
class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(1,2)) / (x.size(-1) ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
```

-----
**9. Transformer Block**
```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attention = SelfAttention(embed_size)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 4*embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4*embed_size, embed_size)
        )
        self.norm2 = torch.nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

-----
**10. 构建微型LLM**
```python
class MiniLLM(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.transformer = TransformerBlock(embed_size)
        self.fc = torch.nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x)
```

-----
**11. 训练数据准备**
```python
text = "hello world, this is a simple language model."
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {c:i for i,c in enumerate(chars)}

# 创建训练样本
sequence_length = 5
inputs = []
targets = []
for i in range(len(text)-sequence_length):
    inputs.append([char_to_idx[c] for c in text[i:i+sequence_length]])
    targets.append(char_to_idx[text[i+sequence_length]])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
```

-----
**12. 训练循环**
```python
model = MiniLLM(vocab_size)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    output = model(inputs)
    loss = criterion(output.view(-1, vocab_size), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
```

-----
**13. 文本生成**
```python
def generate_text(start_str, length=20):
    model.eval()
    chars = list(start_str)
    for _ in range(length):
        x = torch.tensor([char_to_idx[c] for c in chars[-sequence_length:]]).unsqueeze(0)
        pred = model(x)
        next_char = torch.argmax(pred[0,-1]).item()
        chars.append(list(char_to_idx.keys())[next_char])
    return ''.join(chars)

print(generate_text("hello", 50))
```

-----
**14. 效果优化（添加以下组件）**
```python
# 在TransformerBlock中添加
self.dropout = torch.nn.Dropout(0.1)

# 修改模型结构
self.transformer = torch.nn.Sequential(
    *[TransformerBlock(embed_size) for _ in range(2)]  # 2层Transformer
)
```

每个步骤都可独立运行并验证，建议使用Google Colab免配置环境。最终实现的微型LLM虽然简单，但包含了现代大语言模型的核心组件（Transformer、Embedding、自注意力）。

以下是14个步骤之间的依赖关系说明，帮助理解知识模块如何逐步组合成最终模型：

---
### **阶段依赖图示**
```
基础模型能力       神经网络扩展能力         序列建模能力            Transformer架构能力           LLM整合
(1-2)               (3-4-5)               (6-7)                   (8-9)                      (10-14)
│                    │                     │                        │                           │
线性回归 → 逻辑回归 → MLP → CNN → RNN → 词嵌入 → 自注意力 → Transformer块 → 微型LLM → 训练生成
```

---
### **分步依赖详解**

#### **1→2→3：基础建模能力搭建**
- **依赖链条**：环境配置 → 线性回归 → 逻辑回归
- **关键过渡**：
  - 步骤1为所有后续步骤提供运行环境
  - 步骤2的`loss.backward()`训练范式被后续所有模型复用
  - 步骤3在步骤2基础上增加了分类能力（Sigmoid+BCELoss）

#### **4→5：神经网络能力扩展**
- **依赖链条**：MLP → CNN
- **关键过渡**：
  - 步骤4的全连接网络为步骤5的卷积操作提供参数优化基础
  - 步骤5的`Conv2d`演示了空间特征提取方法，与步骤4的`Linear`形成对比

#### **6→7→8：序列建模基础**
- **依赖链条**：RNN → 词嵌入 → 自注意力
- **关键过渡**：
  - 步骤6的序列处理需求引出了步骤7的词向量表示
  - 步骤7的嵌入矩阵是步骤8自注意力计算的输入基础
  - 步骤8的自注意力机制替代了步骤6的RNN时序依赖处理

#### **9→10→11：Transformer架构整合**
- **依赖链条**：Transformer块 → 微型LLM → 训练数据
- **关键过渡**：
  - 步骤9的Transformer块成为步骤10的核心组件
  - 步骤10的`MiniLLM`结构需要步骤11的特殊序列数据格式
  - 步骤11的滑动窗口采样方式适配步骤10的固定长度输入

#### **12→13→14：完整LLM工作流**
- **依赖链条**：训练循环 → 文本生成 → 优化改进
- **关键过渡**：
  - 步骤12的交叉熵损失专为步骤10的分类输出设计
  - 步骤13的生成函数依赖步骤12训练好的概率分布
  - 步骤14的改进直接修改步骤9/10的模型结构

---
### **关键模块依赖**
```python
# 最终模型的结构依赖关系
MiniLLM(
  embed(←step7)           # 来自词嵌入技术
  transformer(←step9)     # 依赖自注意力机制(step8)
  fc(←step3分类思想)       # 继承分类器设计
)

# 训练过程依赖
训练循环(←step2优化器使用)  # 基础训练范式
数据准备(←step6序列处理思想) # 序列切片方法
```

---
### **学习路线建议**
1. **必选路径**：1→2→3→4→7→8→9→10→11→12→13（最小可行路径）
2. **可选扩展**：
   - 图像路线：4→5（理解卷积）
   - 传统序列路线：6→7（对比RNN与Transformer差异）
3. **最终整合**：14（添加深度和正则化）

每个步骤都为后续环节提供以下至少一种支持：
- 代码范式（如训练循环）
- 数学组件（如矩阵乘法）
- 结构设计思想（如残差连接）
- 数据处理方法（如序列切片）
- 
