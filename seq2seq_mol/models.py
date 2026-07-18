"""Neural sequence-to-sequence models used by the molecular translation task."""

import math

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden):
        output, hidden = self.gru(self.embedding(input_ids), hidden)
        return self.output_projection(output), hidden


class Seq2SeqGRU(nn.Module):
    """Teacher-forced GRU encoder-decoder without attention."""

    def __init__(self, encoder, decoder, pad_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(self, src_ids, tgt_ids, src_lengths=None):
        hidden = self.encoder(src_ids, src_lengths)
        logits, _ = self.decoder(tgt_ids[:, :-1], hidden)
        return logits

    def encode(self, src_ids, src_lengths):
        # The final layer's state is the fixed-dimensional molecular embedding.
        return self.encoder(src_ids, src_lengths)[-1]

    @torch.no_grad()
    def generate(self, src_ids, src_lengths, bos_id, eos_id, max_length=128):
        hidden = self.encoder(src_ids, src_lengths)
        next_ids = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        generated = []
        for _ in range(max_length):
            logits, hidden = self.decoder(next_ids[:, -1:], hidden)
            token = logits[:, -1].argmax(dim=-1)
            generated.append(token)
            finished |= token.eq(eos_id)
            next_ids = torch.cat([next_ids, token.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return torch.stack(generated, dim=1) if generated else next_ids[:, :0]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=2048):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_size)
        )
        encoding = torch.zeros(max_length, embed_size)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values):
        return values + self.encoding[:, : values.size(1)]


class TransformerSeq2Seq(nn.Module):
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
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)
        self.position = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(embed_size, tgt_vocab_size)
        self.pad_id = pad_id

    def _embed_source(self, src_ids):
        return self.position(self.src_embedding(src_ids) * math.sqrt(self.src_embedding.embedding_dim))

    def _embed_target(self, tgt_ids):
        return self.position(self.tgt_embedding(tgt_ids) * math.sqrt(self.tgt_embedding.embedding_dim))

    def _memory(self, src_ids):
        source_padding = src_ids.eq(self.pad_id)
        return self.transformer.encoder(self._embed_source(src_ids), src_key_padding_mask=source_padding), source_padding

    def forward(self, src_ids, tgt_ids, src_lengths=None):
        del src_lengths
        tgt_input = tgt_ids[:, :-1]
        memory, source_padding = self._memory(src_ids)
        target_padding = tgt_input.eq(self.pad_id)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=src_ids.device)
        output = self.transformer.decoder(
            self._embed_target(tgt_input),
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output_projection(output)

    def encode(self, src_ids, src_lengths=None):
        del src_lengths
        memory, padding = self._memory(src_ids)
        valid = (~padding).unsqueeze(-1)
        return (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def generate(self, src_ids, src_lengths, bos_id, eos_id, max_length=128):
        del src_lengths
        memory, source_padding = self._memory(src_ids)
        generated = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        for _ in range(max_length):
            target_padding = generated.eq(self.pad_id)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=src_ids.device)
            output = self.transformer.decoder(
                self._embed_target(generated), memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding,
            )
            token = self.output_projection(output[:, -1]).argmax(dim=-1)
            generated = torch.cat([generated, token.unsqueeze(1)], dim=1)
            finished |= token.eq(eos_id)
            if finished.all():
                break
        return generated[:, 1:]
