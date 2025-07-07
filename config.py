class Config:
    # 模型参数
    d_model = 512      # 模型维度
    num_heads = 8      # 多头注意力头数
    dff = 2048         # 前馈网络隐藏层维度
    num_layers = 6     # 编码器层数
    dropout = 0.1      # Dropout率
    
    # 训练参数
    batch_size = 64    # 批大小
    max_seq_len = 100  # 序列最大长度
    epochs = 500        # 训练轮数
    lr = 0.001        # 学习率
    
    # 数据参数
    vocab_size = 10000 # 词汇表大小
