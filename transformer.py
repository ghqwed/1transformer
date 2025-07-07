import torch
import torch.nn as nn
import math

"""
多头注意力机制 (Multi-Head Attention)
论文核心组件，允许模型同时关注不同位置的表示子空间
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层
        
        参数:
            d_model: 输入和输出的维度
            num_heads: 注意力头的数量
        """
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数量
        # 计算每个头的维度
        self.head_dim = d_model // num_heads
        
        # 确保维度可被头数整除
        assert self.head_dim * num_heads == d_model, "d_model 必须能被 num_heads 整除"
        
        # 线性变换层：查询(Q)、键(K)、值(V)
        self.wq = nn.Linear(d_model, d_model)  # 查询变换
        self.wk = nn.Linear(d_model, d_model)  # 键变换
        self.wv = nn.Linear(d_model, d_model)  # 值变换
        # 最终输出线性层
        self.dense = nn.Linear(d_model, d_model)  # 输出变换
    
    def split_heads(self, x, batch_size):
        """
        分割输入张量为多个注意力头
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            batch_size: 批次大小
            
        返回:
            分割后的张量 [batch_size, num_heads, seq_len, head_dim]
        """
        # 重塑张量形状: [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        # 置换维度: [batch_size, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播计算多头注意力
        
        参数:
            q: 查询张量 [batch_size, seq_len_q, d_model]
            k: 键张量 [batch_size, seq_len_k, d_model]
            v: 值张量 [batch_size, seq_len_v, d_model]
            mask: 可选的掩码张量 [batch_size, 1, seq_len_q, seq_len_k]
            
        返回:
            多头注意力输出 [batch_size, seq_len_q, d_model]
        """
        batch_size = q.size(0)  # 获取批次大小
        
        # 线性变换: 投影到查询、键、值空间
        q = self.wq(q)  # [batch_size, seq_len_q, d_model]
        k = self.wk(k)  # [batch_size, seq_len_k, d_model]
        v = self.wv(v)  # [batch_size, seq_len_v, d_model]
        
        # 分割为多个头
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # 缩放点积注意力机制的核心公式
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（用于处理填充或未来信息）
        if mask is not None:
            # 将掩码位置的值设为负无穷，softmax后变为0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重: softmax(注意力分数)
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        
        # 注意力权重与值相乘: 加权求和
        output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 合并注意力头
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len_q, num_heads, head_dim]
        # 重塑为原始维度: [batch_size, seq_len_q, d_model]
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        return self.dense(output)

"""
Transformer编码器层
包含多头注意力和前馈神经网络，以及残差连接和层归一化
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        """
        初始化编码器层
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数量
            dff: 前馈网络内部维度
            dropout: dropout概率
        """
        super().__init__()
        # 多头注意力子层
        self.mha = MultiHeadAttention(d_model, num_heads)
        # 前馈神经网络: 两层线性变换 + ReLU激活
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),  # 扩展维度
            nn.ReLU(),                # 非线性激活
            nn.Linear(dff, d_model)   # 降回原始维度
        )
        # 层归一化
        self.layernorm1 = nn.LayerNorm(d_model)  # 注意力后归一化
        self.layernorm2 = nn.LayerNorm(d_model)  # 前馈网络后归一化
        # Dropout防止过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        编码器层前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        返回:
            编码后的表示 [batch_size, seq_len, d_model]
        """
        # 多头注意力子层
        attn_output = self.mha(x, x, x, mask)  # 自注意力: Q=K=V=x
        attn_output = self.dropout(attn_output)  # 应用dropout
        
        # 残差连接 + 层归一化 (Add & Norm)
        out1 = self.layernorm1(x + attn_output)  # 残差连接后归一化
        
        # 前馈神经网络子层
        ffn_output = self.ffn(out1)  # 前向传播
        ffn_output = self.dropout(ffn_output)  # 应用dropout
        
        # 残差连接 + 层归一化 (Add & Norm)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差连接后归一化
        
        return out2
