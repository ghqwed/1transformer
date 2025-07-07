import torch
from transformer import EncoderLayer, MultiHeadAttention
from config import Config

def test_multi_head_attention():
    print("测试多头注意力机制...")
    config = Config()
    batch_size = 2
    seq_len = 10
    d_model = config.d_model
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, config.num_heads)
    
    # 创建测试输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = mha(q, k, v)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model)
    print("✅ 多头注意力测试通过")

def test_encoder_layer():
    print("测试编码器层...")
    config = Config()
    batch_size = 2
    seq_len = 10
    d_model = config.d_model
    
    # 创建编码器层
    encoder_layer = EncoderLayer(
        d_model, 
        config.num_heads, 
        config.dff, 
        config.dropout
    )
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # 前向传播
    output = encoder_layer(x, mask)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model)
    print("✅ 编码器层测试通过")

def test_transformer():
    print("测试完整Transformer模型...")
    config = Config()
    batch_size = 2
    seq_len = config.max_seq_len
    
    # 从train.py导入Transformer模型
    from train import Transformer
    
    # 创建模型
    model = Transformer(config)
    
    # 创建测试输入
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # 前向传播
    outputs = model(inputs, mask)
    
    # 验证输出形状
    assert outputs.shape == (batch_size, seq_len, config.vocab_size)
    print("✅ Transformer模型测试通过")

if __name__ == "__main__":
    test_multi_head_attention()
    test_encoder_layer()
    test_transformer()
    print("所有测试通过！")
