import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    """基于Transformer的Seq2Seq模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout=0.1, max_seq_len=128):
        super(TransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 源语言嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # 目标语言嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch_size, features)
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 特殊token
        self.sos_token = 1  # <sos>
        self.eos_token = 2  # <eos>
        self.pad_token = 0  # <pad>
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        前向传播
        
        参数:
            src: 源序列 (src_seq_len, batch_size)
            tgt: 目标序列 (tgt_seq_len, batch_size)
            src_key_padding_mask: 源序列padding掩码 (batch_size, src_seq_len)
            tgt_key_padding_mask: 目标序列padding掩码 (batch_size, tgt_seq_len)
        """
        # 嵌入层 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoder(src_emb)
        
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoder(tgt_emb)
        
        # 创建目标序列掩码
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        # Transformer前向传播
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层
        output = self.fc_out(output)
        
        return output
    
    def translate(self, src, max_len=128):
        """翻译单个序列（推理模式）"""
        with torch.no_grad():
            # 添加批次维度
            src = src.unsqueeze(1)  # (src_seq_len, 1)
            
            # 嵌入层 + 位置编码
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.src_pos_encoder(src_emb)
            
            # 初始化目标序列
            tgt = torch.tensor([[self.sos_token]], device=src.device)  # (1, 1)
            
            # 逐步生成输出
            outputs = []
            for i in range(max_len):
                # 嵌入层 + 位置编码
                tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = self.tgt_pos_encoder(tgt_emb)
                
                # 创建目标序列掩码
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                
                # Transformer前向传播
                output = self.transformer(
                    src=src_emb,
                    tgt=tgt_emb,
                    tgt_mask=tgt_mask
                )
                
                # 预测下一个token
                output = self.fc_out(output[-1, :, :])
                next_token = output.argmax(1).item()
                outputs.append(next_token)
                
                # 遇到<eos>则停止
                if next_token == self.eos_token:
                    break
                
                # 更新目标序列
                next_token_tensor = torch.tensor([[next_token]], device=src.device)
                tgt = torch.cat([tgt, next_token_tensor], dim=0)
            
            return outputs
    
    def get_molecular_representation(self, src):
        """获取分子表示（编码器输出均值池化）"""
        with torch.no_grad():
            # 添加批次维度
            src = src.unsqueeze(1)  # (src_seq_len, 1)
            
            # 嵌入层 + 位置编码
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.src_pos_encoder(src_emb)
            
            # 编码器前向传播
            encoder_output = self.transformer.encoder(src_emb)
            
            # 均值池化
            representation = torch.mean(encoder_output, dim=0).squeeze(0)
            
            return representation

# 示例用法
if __name__ == "__main__":
    # 模拟参数
    src_vocab_size = 100
    tgt_vocab_size = 150
    d_model = 512
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 2048
    max_seq_len = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = TransformerSeq2Seq(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_len=max_seq_len
    ).to(device)
    
    # 模拟输入
    src_seq = torch.randint(0, src_vocab_size, (20, 32)).to(device)  # (seq_len, batch_size)
    tgt_seq = torch.randint(0, tgt_vocab_size, (25, 32)).to(device)
    
    # 前向传播
    outputs = model(src_seq, tgt_seq)
    print(f"输出形状: {outputs.shape}")  # 应为 (25, 32, 150)
    
    # 测试翻译
    test_src = torch.randint(0, src_vocab_size, (15,)).to(device)
    translation = model.translate(test_src)
    print(f"翻译结果: {translation}")
    
    # 测试分子表示
    representation = model.get_molecular_representation(test_src)
    print(f"分子表示形状: {representation.shape}")  # 应为 (512,)