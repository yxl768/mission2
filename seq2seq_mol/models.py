"""分子翻译任务使用的神经网络序列到序列模型."""

import math

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU编码器 - 将输入序列编码为隐藏状态表示.
    
    使用双向GRU处理输入序列，同时从左到右和从右到左捕捉上下文信息。
    通过PackedSequence处理变长序列，提高计算效率。
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=True):
        """初始化GRU编码器.
        
        Args:
            vocab_size: 输入词汇表大小
            embed_size: 词嵌入维度
            hidden_size: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率（仅在多层时生效）
            bidirectional: 是否使用双向GRU
        """
        super().__init__()
        # 词嵌入层：将token ID转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU层：处理序列输入，生成隐藏状态
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional

    def forward(self, input_ids, lengths):
        """前向传播 - 编码输入序列.
        
        Args:
            input_ids: 输入序列的token ID，形状 (batch_size, max_len)
            lengths: 每个序列的真实长度，形状 (batch_size,)
        
        Returns:
            output: 所有时刻的隐藏状态，形状 (batch_size, max_len, hidden_size * num_directions)
            hidden: 最后时刻的隐藏状态，形状 (num_layers * num_directions, batch_size, hidden_size)
        """
        # 将token ID转换为词嵌入
        embedded = self.embedding(input_ids)
        # 打包序列，跳过padding部分
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # GRU前向传播
        output, hidden = self.gru(packed)
        # 解包序列，恢复padding
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制 - 加性注意力.
    
    在解码过程中，动态关注编码器输出的不同部分，生成上下文向量。
    适用于编码器和解码器隐藏维度不同的情况。
    """

    def __init__(self, hidden_size, enc_hidden_size):
        """初始化Bahdanau注意力.
        
        Args:
            hidden_size: 解码器隐藏状态维度
            enc_hidden_size: 编码器隐藏状态维度
        """
        super().__init__()
        # 编码器输出变换矩阵
        self.W_a = nn.Linear(enc_hidden_size, hidden_size, bias=False)
        # 解码器隐藏状态变换矩阵
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        # 注意力打分矩阵
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, padding_mask=None):
        """计算注意力权重和上下文向量.
        
        Args:
            decoder_hidden: 当前解码器隐藏状态，形状 (batch_size, hidden_size)
            encoder_outputs: 编码器所有时刻输出，形状 (batch_size, seq_len, enc_hidden_size)
            padding_mask: padding位置掩码，形状 (batch_size, seq_len)
        
        Returns:
            context: 上下文向量，形状 (batch_size, enc_hidden_size)
            weights: 注意力权重，形状 (batch_size, seq_len, 1)
        """
        # 扩展解码器隐藏状态维度，便于与编码器输出计算
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        # 计算注意力分数：score = v_a^T * tanh(W_a * encoder_outputs + U_a * decoder_hidden)
        scores = self.v_a(torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden_expanded)))
        # 对padding位置设置负无穷，使其在softmax中权重为0
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        # softmax归一化得到注意力权重
        weights = torch.softmax(scores, dim=1)
        # 加权求和得到上下文向量
        context = (encoder_outputs * weights).sum(dim=1)
        return context, weights


class GRUDecoder(nn.Module):
    """GRU解码器 - 根据上下文向量生成目标序列.
    
    在每个解码步骤，结合当前token嵌入和注意力上下文向量，预测下一个token。
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        """初始化GRU解码器.
        
        Args:
            vocab_size: 目标词汇表大小
            embed_size: 词嵌入维度
            hidden_size: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU层：输入维度为 embed_size + hidden_size（拼接词嵌入和上下文向量）
        self.gru = nn.GRU(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 输出投影层：将GRU输出和上下文向量的拼接投影到词汇表
        self.output_projection = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_ids, hidden, context):
        """前向传播 - 单步解码.
        
        Args:
            input_ids: 当前输入token ID，形状 (batch_size, 1)
            hidden: 解码器隐藏状态，形状 (num_layers, batch_size, hidden_size)
            context: 注意力上下文向量，形状 (batch_size, hidden_size)
        
        Returns:
            logits: 词汇表上的概率分布，形状 (batch_size, 1, vocab_size)
            hidden: 更新后的解码器隐藏状态
        """
        # 将token ID转换为词嵌入
        embedded = self.embedding(input_ids)
        # 拼接词嵌入和上下文向量作为GRU输入
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        # GRU前向传播
        output, hidden = self.gru(gru_input, hidden)
        # 拼接GRU输出和上下文向量，增强表达能力
        combined = torch.cat([output, context.unsqueeze(1)], dim=-1)
        # 投影到词汇表空间
        return self.output_projection(combined), hidden


class Seq2SeqGRU(nn.Module):
    """带Bahdanau注意力的GRU编码器-解码器模型.
    
    实现了完整的序列到序列翻译功能，支持教师强制训练、贪婪搜索生成和束搜索生成。
    """

    def __init__(self, encoder, decoder, pad_id):
        """初始化Seq2SeqGRU模型.
        
        Args:
            encoder: GRU编码器实例
            decoder: GRU解码器实例
            pad_id: padding token的ID
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        # 计算编码器隐藏状态维度（双向时加倍）
        enc_hidden_size = encoder.gru.hidden_size * (2 if encoder.bidirectional else 1)
        # 初始化Bahdanau注意力模块
        self.attention = BahdanauAttention(decoder.gru.hidden_size, enc_hidden_size)
        # 双向GRU时需要投影层将2*hidden_size映射到decoder的hidden_size
        if encoder.bidirectional:
            self.hidden_projection = nn.Linear(encoder.gru.hidden_size * 2, decoder.gru.hidden_size)
        else:
            self.hidden_projection = None

    def _combine_bidirectional_hidden(self, hidden):
        """合并双向GRU的隐藏状态.
        
        将双向GRU的前向和后向隐藏状态拼接，并投影到解码器所需维度。
        
        Args:
            hidden: 双向GRU的隐藏状态，形状 (2*num_layers, batch_size, hidden_size)
        
        Returns:
            combined: 合并后的隐藏状态，形状 (num_layers, batch_size, decoder_hidden_size)
        """
        if self.encoder.bidirectional:
            num_layers = hidden.size(0) // 2
            # 调整形状：(2*num_layers, batch, hidden) -> (num_layers, 2, batch, hidden)
            hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
            forward = hidden[:, 0]  # 前向GRU隐藏状态
            backward = hidden[:, 1]  # 后向GRU隐藏状态
            # 拼接前向和后向
            combined = torch.cat([forward, backward], dim=-1)
            # 投影到解码器维度
            if self.hidden_projection is not None:
                combined = self.hidden_projection(combined)
            return combined
        return hidden

    def _combine_bidirectional_output(self, output):
        """合并双向GRU的输出序列.
        
        Args:
            output: 双向GRU的输出，形状 (batch_size, seq_len, 2*hidden_size)
        
        Returns:
            combined: 合并后的输出，形状 (batch_size, seq_len, 2*hidden_size)
        """
        if self.encoder.bidirectional:
            batch_size, seq_len, _ = output.size()
            # 调整形状并拼接
            output = output.view(batch_size, seq_len, 2, self.encoder.gru.hidden_size)
            return torch.cat([output[:, :, 0], output[:, :, 1]], dim=-1)
        return output

    def forward(self, src_ids, tgt_ids, src_lengths=None):
        """前向传播 - 教师强制训练模式.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            tgt_ids: 目标序列token ID，形状 (batch_size, tgt_len)
            src_lengths: 源序列真实长度，形状 (batch_size,)
        
        Returns:
            logits: 目标序列每个位置的预测概率，形状 (batch_size, tgt_len-1, vocab_size)
        """
        # 编码源序列
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        # 合并双向输出
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        # 合并双向隐藏状态作为解码器初始状态
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        # 生成padding掩码
        padding_mask = src_ids.eq(self.pad_id)

        # 逐步骤解码（教师强制）
        logits_list = []
        # 计算初始注意力上下文向量
        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        for i in range(tgt_ids.size(1) - 1):
            # 取第i个token作为输入
            input_token = tgt_ids[:, i:i+1]
            # 单步解码
            step_logits, decoder_hidden = self.decoder(input_token, decoder_hidden, context)
            logits_list.append(step_logits)
            # 更新注意力上下文向量
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)

        # 拼接所有步骤的输出
        return torch.cat(logits_list, dim=1)

    def encode(self, src_ids, src_lengths):
        """提取分子表示 - 将IUPAC名称编码为固定长度向量.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度，形状 (batch_size,)
        
        Returns:
            embedding: 分子表示向量，形状 (batch_size, 2*hidden_size)
        """
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        if self.encoder.bidirectional:
            # 取双向GRU最后两层隐藏状态
            final_hidden = encoder_hidden[-2:]
            # 拼接前向和后向隐藏状态
            combined = torch.cat([final_hidden[0], final_hidden[1]], dim=-1)
            return combined
        return encoder_hidden[-1]

    @torch.no_grad()
    def generate(self, src_ids, src_lengths, bos_id, eos_id, max_length=128):
        """贪婪搜索生成目标序列.
        
        在每个步骤选择概率最大的token作为下一个输入。
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度，形状 (batch_size,)
            bos_id: 开始符token ID
            eos_id: 结束符token ID
            max_length: 最大生成长度
        
        Returns:
            generated: 生成的目标序列，形状 (batch_size, gen_len)
        """
        # 编码源序列
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        padding_mask = src_ids.eq(self.pad_id)

        # 初始化生成序列，以bos token开头
        next_ids = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        generated = []

        # 计算初始上下文向量
        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        for _ in range(max_length):
            # 单步解码
            logits, decoder_hidden = self.decoder(next_ids[:, -1:], decoder_hidden, context)
            # 选择概率最大的token
            token = logits[:, -1].argmax(dim=-1)
            generated.append(token)
            # 标记已完成的序列
            finished |= token.eq(eos_id)
            next_ids = torch.cat([next_ids, token.unsqueeze(1)], dim=1)
            # 所有序列都完成则停止
            if finished.all():
                break
            # 更新上下文向量
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)

        return torch.stack(generated, dim=1) if generated else next_ids[:, :0]

    @torch.no_grad()
    def generate_beam_search(self, src_ids, src_lengths, bos_id, eos_id, max_length=128, beam_size=5):
        """束搜索生成目标序列.
        
        维护beam_size个候选序列，避免贪婪搜索的局部最优问题，生成质量更高。
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度，形状 (batch_size,)
            bos_id: 开始符token ID
            eos_id: 结束符token ID
            max_length: 最大生成长度
            beam_size: 束搜索宽度
        
        Returns:
            best_sequence: 最优生成序列，形状 (batch_size, gen_len)
        """
        # 编码源序列
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        padding_mask = src_ids.eq(self.pad_id)

        batch_size = src_ids.size(0)
        # 初始化beam_size个候选序列
        sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        scores = torch.zeros(batch_size, beam_size, device=src_ids.device)
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src_ids.device)

        # 扩展上下文向量和解码器隐藏状态以支持beam搜索
        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        context = context.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
        decoder_hidden = decoder_hidden.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(-1, batch_size * beam_size, -1)

        for _ in range(max_length):
            # 获取每个候选序列的最后一个token
            input_tokens = sequences[:, :, -1].reshape(batch_size * beam_size, 1)
            # 单步解码
            logits, decoder_hidden = self.decoder(input_tokens, decoder_hidden, context)
            # 计算对数概率
            log_probs = torch.log_softmax(logits[:, -1], dim=-1)

            # 调整形状用于beam搜索
            log_probs = log_probs.view(batch_size, beam_size, -1)
            # 已完成序列的概率设为负无穷
            log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
            # 累积分数
            cum_scores = scores.unsqueeze(-1) + log_probs

            # 选择top-k候选
            top_scores, top_indices = cum_scores.view(batch_size, -1).topk(beam_size, dim=-1)
            beam_indices = top_indices // log_probs.size(-1)  # 来自哪个beam
            token_indices = top_indices % log_probs.size(-1)   # 选择了哪个token

            # 更新候选序列
            new_sequences = []
            for b in range(batch_size):
                new_seq = []
                for k in range(beam_size):
                    new_seq.append(sequences[b, beam_indices[b, k]].clone())
                new_sequences.append(torch.stack(new_seq))
            sequences = torch.stack(new_sequences)
            sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)

            # 更新分数和完成状态
            scores = top_scores
            finished = torch.gather(finished, 1, beam_indices) | token_indices.eq(eos_id)

            # 更新上下文向量和解码器隐藏状态
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs.repeat_interleave(beam_size, dim=1), padding_mask.repeat_interleave(beam_size, dim=0))
            decoder_hidden = decoder_hidden.view(-1, batch_size, beam_size, decoder_hidden.size(-1))
            decoder_hidden = torch.gather(decoder_hidden, 2, beam_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, decoder_hidden.size(-1)))
            decoder_hidden = decoder_hidden.reshape(-1, batch_size * beam_size, -1)

            # 所有序列都完成则停止
            if finished.all():
                break

        # 选择分数最高的序列
        best_indices = scores.argmax(dim=-1)
        result = []
        for b in range(batch_size):
            result.append(sequences[b, best_indices[b], 1:])  # 去掉bos token
        return torch.cat([r.unsqueeze(0) for r in result], dim=0)


class PositionalEncoding(nn.Module):
    """位置编码 - 为Transformer提供序列位置信息.
    
    使用正弦/余弦函数为每个位置生成唯一的编码，使模型能够感知序列顺序。
    """

    def __init__(self, embed_size, max_length=2048):
        """初始化位置编码.
        
        Args:
            embed_size: 嵌入维度
            max_length: 最大序列长度
        """
        super().__init__()
        # 生成位置索引
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # 生成频率参数
        frequencies = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_size)
        )
        # 计算位置编码
        encoding = torch.zeros(max_length, embed_size)
        encoding[:, 0::2] = torch.sin(position * frequencies)  # 偶数位置用正弦
        encoding[:, 1::2] = torch.cos(position * frequencies)  # 奇数位置用余弦
        # 注册为缓冲区（不参与训练）
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values):
        """前向传播 - 添加位置编码.
        
        Args:
            values: 输入嵌入向量，形状 (batch_size, seq_len, embed_size)
        
        Returns:
            output: 添加位置编码后的向量
        """
        return values + self.encoding[:, : values.size(1)]


class TransformerSeq2Seq(nn.Module):
    """Transformer序列到序列模型.
    
    使用多头自注意力机制建模序列依赖关系，支持并行计算，性能通常优于GRU模型。
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_size=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=0,
    ):
        """初始化TransformerSeq2Seq模型.
        
        Args:
            src_vocab_size: 源词汇表大小
            tgt_vocab_size: 目标词汇表大小
            embed_size: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout概率
            pad_id: padding token的ID
        """
        super().__init__()
        # 源序列词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_id)
        # 目标序列词嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)
        # 位置编码层
        self.position = PositionalEncoding(embed_size)
        # Transformer核心组件
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # 输出投影层
        self.output_projection = nn.Linear(embed_size, tgt_vocab_size)
        self.pad_id = pad_id

    def _embed_source(self, src_ids):
        """嵌入源序列 - 添加词嵌入和位置编码.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
        
        Returns:
            embedded: 嵌入后的序列，形状 (batch_size, src_len, embed_size)
        """
        return self.position(self.src_embedding(src_ids) * math.sqrt(self.src_embedding.embedding_dim))

    def _embed_target(self, tgt_ids):
        """嵌入目标序列 - 添加词嵌入和位置编码.
        
        Args:
            tgt_ids: 目标序列token ID，形状 (batch_size, tgt_len)
        
        Returns:
            embedded: 嵌入后的序列，形状 (batch_size, tgt_len, embed_size)
        """
        return self.position(self.tgt_embedding(tgt_ids) * math.sqrt(self.tgt_embedding.embedding_dim))

    def _memory(self, src_ids):
        """编码源序列得到memory表示.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
        
        Returns:
            memory: 编码器输出，形状 (batch_size, src_len, embed_size)
            source_padding: padding掩码，形状 (batch_size, src_len)
        """
        source_padding = src_ids.eq(self.pad_id)
        return self.transformer.encoder(self._embed_source(src_ids), src_key_padding_mask=source_padding), source_padding

    def forward(self, src_ids, tgt_ids, src_lengths=None):
        """前向传播 - 教师强制训练模式.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            tgt_ids: 目标序列token ID，形状 (batch_size, tgt_len)
            src_lengths: 源序列真实长度（Transformer不需要）
        
        Returns:
            logits: 目标序列每个位置的预测概率，形状 (batch_size, tgt_len-1, vocab_size)
        """
        del src_lengths  # Transformer不需要序列长度信息
        # 目标序列去掉最后一个token作为输入
        tgt_input = tgt_ids[:, :-1]
        # 编码源序列
        memory, source_padding = self._memory(src_ids)
        # 生成目标序列padding掩码
        target_padding = tgt_input.eq(self.pad_id)
        # 生成因果掩码（防止解码器看到未来token）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=src_ids.device)
        # Transformer解码器前向传播
        output = self.transformer.decoder(
            self._embed_target(tgt_input),
            memory,
            tgt_mask=causal_mask,                  # 因果掩码
            tgt_key_padding_mask=target_padding,    # 目标padding掩码
            memory_key_padding_mask=source_padding, # 源padding掩码
        )
        # 投影到词汇表空间
        return self.output_projection(output)

    def encode(self, src_ids, src_lengths=None):
        """提取分子表示 - 将IUPAC名称编码为固定长度向量.
        
        通过均值池化编码器输出，得到固定长度的分子表示。
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度（Transformer不需要）
        
        Returns:
            embedding: 分子表示向量，形状 (batch_size, embed_size)
        """
        del src_lengths
        memory, padding = self._memory(src_ids)
        # 创建有效位置掩码
        valid = (~padding).unsqueeze(-1)
        # 均值池化：只考虑非padding位置
        return (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def generate(self, src_ids, src_lengths, bos_id, eos_id, max_length=128):
        """贪婪搜索生成目标序列.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度
            bos_id: 开始符token ID
            eos_id: 结束符token ID
            max_length: 最大生成长度
        
        Returns:
            generated: 生成的目标序列，形状 (batch_size, gen_len)
        """
        del src_lengths
        # 编码源序列
        memory, source_padding = self._memory(src_ids)
        # 初始化生成序列
        generated = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        
        for _ in range(max_length):
            # 生成padding掩码和因果掩码
            target_padding = generated.eq(self.pad_id)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=src_ids.device)
            # Transformer解码器前向传播
            output = self.transformer.decoder(
                self._embed_target(generated), memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding,
            )
            # 选择概率最大的token
            token = self.output_projection(output[:, -1]).argmax(dim=-1)
            generated = torch.cat([generated, token.unsqueeze(1)], dim=1)
            # 标记已完成的序列
            finished |= token.eq(eos_id)
            if finished.all():
                break
        
        return generated[:, 1:]  # 去掉bos token

    @torch.no_grad()
    def generate_beam_search(self, src_ids, src_lengths, bos_id, eos_id, max_length=128, beam_size=5):
        """束搜索生成目标序列.
        
        Args:
            src_ids: 源序列token ID，形状 (batch_size, src_len)
            src_lengths: 源序列真实长度
            bos_id: 开始符token ID
            eos_id: 结束符token ID
            max_length: 最大生成长度
            beam_size: 束搜索宽度
        
        Returns:
            best_sequence: 最优生成序列，形状 (batch_size, gen_len)
        """
        del src_lengths
        # 编码源序列
        memory, source_padding = self._memory(src_ids)
        batch_size = src_ids.size(0)

        # 初始化beam_size个候选序列
        sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        scores = torch.zeros(batch_size, beam_size, device=src_ids.device)
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src_ids.device)

        # 扩展memory和padding以支持beam搜索
        memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(batch_size * beam_size, memory.size(1), memory.size(2))
        source_padding = source_padding.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)

        for _ in range(max_length):
            # 获取所有候选序列
            input_tokens = sequences.reshape(batch_size * beam_size, -1)
            # 生成padding掩码和因果掩码
            target_padding = input_tokens.eq(self.pad_id)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(input_tokens.size(1), device=src_ids.device)

            # Transformer解码器前向传播
            output = self.transformer.decoder(
                self._embed_target(input_tokens), memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding,
            )
            # 计算对数概率
            logits = self.output_projection(output[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # 调整形状用于beam搜索
            log_probs = log_probs.view(batch_size, beam_size, -1)
            log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
            cum_scores = scores.unsqueeze(-1) + log_probs

            # 选择top-k候选
            top_scores, top_indices = cum_scores.view(batch_size, -1).topk(beam_size, dim=-1)
            beam_indices = top_indices // log_probs.size(-1)
            token_indices = top_indices % log_probs.size(-1)

            # 更新候选序列
            new_sequences = []
            for b in range(batch_size):
                new_seq = []
                for k in range(beam_size):
                    new_seq.append(sequences[b, beam_indices[b, k]].clone())
                new_sequences.append(torch.stack(new_seq))
            sequences = torch.stack(new_sequences)
            sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)

            # 更新分数和完成状态
            scores = top_scores
            finished = torch.gather(finished, 1, beam_indices) | token_indices.eq(eos_id)

            # 所有序列都完成则停止
            if finished.all():
                break

        # 选择分数最高的序列
        best_indices = scores.argmax(dim=-1)
        result = []
        for b in range(batch_size):
            result.append(sequences[b, best_indices[b], 1:])  # 去掉bos token
        return torch.cat([r.unsqueeze(0) for r in result], dim=0)
