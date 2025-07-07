import torch
import torch.nn as nn
import torch.optim as optim
from transformer import EncoderLayer, MultiHeadAttention
from config import Config
import numpy as np

"""
Transformer模型训练脚本
使用合成数据训练简化版Transformer模型
"""

# 生成合成数据函数
def generate_data(batch_size, seq_len, vocab_size):
    """
    生成随机训练数据
    
    参数:
        batch_size: 批次大小
        seq_len: 序列长度
        vocab_size: 词汇表大小
        
    返回:
        data: 随机整数序列 [batch_size, seq_len]
        mask: 注意力掩码（全1表示无遮挡）[batch_size, 1, seq_len, seq_len]
    """
    # 创建随机整数序列，范围[0, vocab_size)
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 创建全1注意力掩码（表示所有位置都是有效的）
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    return data, mask

# 简化版Transformer模型类
class Transformer(nn.Module):
    """
    简化版Transformer模型
    包含词嵌入层、多个编码器层和最后的线性分类层
    """
    def __init__(self, config):
        """
        初始化Transformer模型
        
        参数:
            config: 包含模型配置的对象
        """
        super().__init__()
        # 词嵌入层: 将整数索引转换为密集向量
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 创建多个编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.num_heads, config.dff, config.dropout)
            for _ in range(config.num_layers)  # 重复创建num_layers个编码器层
        ])
        
        # 最后的全连接层: 将编码器输出转换为词汇表大小的logits
        self.fc = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, x, mask):
        """
        模型前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]
            
        返回:
            预测logits [batch_size, seq_len, vocab_size]
        """
        # 将输入序列转换为嵌入向量
        x = self.embedding(x)
        
        # 通过每个编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)
            
        # 通过全连接层生成预测
        return self.fc(x)

# 训练函数
def train():
    """
    主训练函数
    配置模型、优化器，运行训练循环并保存模型
    """
    config = Config()  # 加载配置
    
    # 设置设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型并移到设备上
    model = Transformer(config).to(device)
    
    # 定义损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器（Adam优化器）
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    print(f"开始训练，共 {config.epochs} 个epoch...")
    
    # 训练循环 - 每个epoch处理一批数据
    for epoch in range(config.epochs):
        # 生成合成数据
        inputs, mask = generate_data(
            config.batch_size, 
            config.max_seq_len, 
            config.vocab_size
        )
        # 将数据移到设备上
        inputs, mask = inputs.to(device), mask.to(device)
        
        # 模型前向传播
        outputs = model(inputs, mask)
        
        # 计算损失: 将输出展平为[batch_size*seq_len, vocab_size]
        # 目标标签展平为[batch_size*seq_len]
        loss = criterion(outputs.view(-1, config.vocab_size), inputs.view(-1))
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重
        
        # 打印训练进度
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {loss.item():.4f}")
    
    # 训练完成后保存模型权重
    torch.save(model.state_dict(), "transformer_model.pth")
    print("训练完成，模型权重已保存到 transformer_model.pth")

if __name__ == "__main__":
    train()
