import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    """基于GRU的编码器"""
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # 双向GRU
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 将双向输出转换为单隐藏状态
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        # x: (seq_len, batch_size)
        embedding = self.embedding(x)
        # embedding: (seq_len, batch_size, embedding_dim)
        
        # 初始化隐藏状态
        batch_size = x.shape[1]
        hidden = self.init_hidden(batch_size, x.device)
        
        # GRU前向传播
        outputs, hidden = self.gru(embedding, hidden)
        # outputs: (seq_len, batch_size, hidden_size*2)
        # hidden: (num_layers*2, batch_size, hidden_size)
        
        # 合并双向GRU的最终隐藏状态
        hidden_forward = hidden[-2, :, :]  # 前向最后层
        hidden_backward = hidden[-1, :, :]  # 后向最后层
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # 转换到单隐藏状态
        hidden_final = self.fc_hidden(hidden_concat).unsqueeze(0)
        
        return outputs, hidden_final
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return hidden

class Attention(nn.Module):
    """Bahdanau注意力机制"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size*2)
        
        # 扩展decoder_hidden以匹配encoder_outputs的时间步
        decoder_hidden = decoder_hidden.repeat(encoder_outputs.shape[0], 1, 1)
        
        # 计算注意力分数
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))
        attention_scores = self.V(energy).squeeze(2)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=0).unsqueeze(2)
        
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=0)
        
        return context_vector, attention_weights

class DecoderGRU(nn.Module):
    """基于GRU的解码器（带注意力）"""
    def __init__(self, output_size, embedding_dim, hidden_size, num_layers=1, dropout=0.3):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 嵌入层
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # 注意力机制
        self.attention = Attention(hidden_size)
        
        # GRU层
        self.gru = nn.GRU(
            embedding_dim + hidden_size * 2,  # 输入: 嵌入 + 上下文向量
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size * 3 + embedding_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, encoder_outputs):
        # x: (batch_size)
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size*2)
        
        # 添加序列维度
        x = x.unsqueeze(0)
        # x: (1, batch_size)
        
        # 嵌入层
        embedding = self.dropout(self.embedding(x))
        # embedding: (1, batch_size, embedding_dim)
        
        # 计算注意力
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)
        context_vector = context_vector.unsqueeze(0)
        # context_vector: (1, batch_size, hidden_size*2)
        
        # GRU输入: 嵌入 + 上下文向量
        gru_input = torch.cat((embedding, context_vector), dim=2)
        # gru_input: (1, batch_size, embedding_dim + hidden_size*2)
        
        # GRU前向传播
        output, hidden = self.gru(gru_input, hidden)
        # output: (1, batch_size, hidden_size)
        
        # 准备全连接层输入
        output = torch.cat((output.squeeze(0), context_vector.squeeze(0), embedding.squeeze(0)), dim=1)
        # output: (batch_size, hidden_size + hidden_size*2 + embedding_dim)
        
        # 预测输出token
        prediction = self.fc_out(output)
        
        return prediction, hidden, attention_weights

class Seq2SeqGRU(nn.Module):
    """完整的Seq2Seq模型（GRU版本）"""
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (src_seq_len, batch_size)
        # tgt: (tgt_seq_len, batch_size)
        
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_size
        
        # 存储解码器输出
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 初始解码器输入: <sos> token
        decoder_input = tgt[0, :]
        
        # 解码器逐步生成输出
        for t in range(1, tgt_len):
            # 解码器前向传播
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 存储输出
            outputs[t] = output
            
            # 决定下一个输入: 真实值或预测值
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[t] if teacher_force else top1
        
        return outputs
    
    def translate(self, src, max_len=128):
        """翻译单个序列（推理模式）"""
        with torch.no_grad():
            # 添加批次维度
            src = src.unsqueeze(1)
            
            # 编码器前向传播
            encoder_outputs, hidden = self.encoder(src)
            
            # 初始化解码器输入
            decoder_input = torch.tensor([1], device=self.device)  # <sos> token
            
            # 存储输出序列
            outputs = []
            attentions = []
            
            # 逐步生成输出
            for t in range(max_len):
                output, hidden, attention = self.decoder(decoder_input, hidden, encoder_outputs)
                
                # 获取预测token
                top1 = output.argmax(1)
                outputs.append(top1.item())
                
                # 存储注意力权重
                attentions.append(attention.squeeze().cpu().numpy())
                
                # 遇到<eos>则停止
                if top1.item() == 2:  # <eos> token
                    break
                
                # 下一个输入为当前预测
                decoder_input = top1
            
            return outputs, attentions
    
    def get_molecular_representation(self, src):
        """获取分子表示（编码器最终隐藏状态）"""
        with torch.no_grad():
            # 添加批次维度
            src = src.unsqueeze(1)
            
            # 编码器前向传播
            _, hidden = self.encoder(src)
            
            # 返回分子表示
            return hidden.squeeze(0)

# 示例用法
if __name__ == "__main__":
    # 模拟参数
    src_vocab_size = 100
    tgt_vocab_size = 150
    embedding_dim = 256
    hidden_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    encoder = EncoderGRU(src_vocab_size, embedding_dim, hidden_size)
    decoder = DecoderGRU(tgt_vocab_size, embedding_dim, hidden_size)
    model = Seq2SeqGRU(encoder, decoder, device).to(device)
    
    # 模拟输入
    src_seq = torch.randint(0, src_vocab_size, (20, 32)).to(device)  # (seq_len, batch_size)
    tgt_seq = torch.randint(0, tgt_vocab_size, (25, 32)).to(device)
    
    # 前向传播
    outputs = model(src_seq, tgt_seq)
    print(f"输出形状: {outputs.shape}")  # 应为 (25, 32, 150)
    
    # 测试翻译
    test_src = torch.randint(0, src_vocab_size, (15,)).to(device)
    translation, attentions = model.translate(test_src)
    print(f"翻译结果: {translation}")
    
    # 测试分子表示
    representation = model.get_molecular_representation(test_src)
    print(f"分子表示形状: {representation.shape}")  # 应为 (512,)