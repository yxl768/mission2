"""Neural sequence-to-sequence models used by the molecular translation task."""

import math

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
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
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, enc_hidden_size):
        super().__init__()
        self.W_a = nn.Linear(enc_hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, padding_mask=None):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        scores = self.v_a(torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden_expanded)))
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (encoder_outputs * weights).sum(dim=1)
        return context, weights


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_ids, hidden, context):
        embedded = self.embedding(input_ids)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.gru(gru_input, hidden)
        combined = torch.cat([output, context.unsqueeze(1)], dim=-1)
        return self.output_projection(combined), hidden


class Seq2SeqGRU(nn.Module):
    """Teacher-forced GRU encoder-decoder with Bahdanau attention."""

    def __init__(self, encoder, decoder, pad_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        enc_hidden_size = encoder.gru.hidden_size * (2 if encoder.bidirectional else 1)
        self.attention = BahdanauAttention(decoder.gru.hidden_size, enc_hidden_size)
        if encoder.bidirectional:
            self.hidden_projection = nn.Linear(encoder.gru.hidden_size * 2, decoder.gru.hidden_size)
        else:
            self.hidden_projection = None

    def _combine_bidirectional_hidden(self, hidden):
        if self.encoder.bidirectional:
            num_layers = hidden.size(0) // 2
            hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
            forward = hidden[:, 0]
            backward = hidden[:, 1]
            combined = torch.cat([forward, backward], dim=-1)
            if self.hidden_projection is not None:
                combined = self.hidden_projection(combined)
            return combined
        return hidden

    def _combine_bidirectional_output(self, output):
        if self.encoder.bidirectional:
            batch_size, seq_len, _ = output.size()
            output = output.view(batch_size, seq_len, 2, self.encoder.gru.hidden_size)
            return torch.cat([output[:, :, 0], output[:, :, 1]], dim=-1)
        return output

    def forward(self, src_ids, tgt_ids, src_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        padding_mask = src_ids.eq(self.pad_id)

        logits_list = []
        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        for i in range(tgt_ids.size(1) - 1):
            input_token = tgt_ids[:, i:i+1]
            step_logits, decoder_hidden = self.decoder(input_token, decoder_hidden, context)
            logits_list.append(step_logits)
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)

        return torch.cat(logits_list, dim=1)

    def encode(self, src_ids, src_lengths):
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        if self.encoder.bidirectional:
            final_hidden = encoder_hidden[-2:]
            combined = torch.cat([final_hidden[0], final_hidden[1]], dim=-1)
            return combined
        return encoder_hidden[-1]

    @torch.no_grad()
    def generate(self, src_ids, src_lengths, bos_id, eos_id, max_length=128):
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        padding_mask = src_ids.eq(self.pad_id)

        next_ids = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        generated = []

        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        for _ in range(max_length):
            logits, decoder_hidden = self.decoder(next_ids[:, -1:], decoder_hidden, context)
            token = logits[:, -1].argmax(dim=-1)
            generated.append(token)
            finished |= token.eq(eos_id)
            next_ids = torch.cat([next_ids, token.unsqueeze(1)], dim=1)
            if finished.all():
                break
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)

        return torch.stack(generated, dim=1) if generated else next_ids[:, :0]

    @torch.no_grad()
    def generate_beam_search(self, src_ids, src_lengths, bos_id, eos_id, max_length=128, beam_size=5):
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        encoder_outputs = self._combine_bidirectional_output(encoder_outputs)
        decoder_hidden = self._combine_bidirectional_hidden(encoder_hidden)
        padding_mask = src_ids.eq(self.pad_id)

        batch_size = src_ids.size(0)
        sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        scores = torch.zeros(batch_size, beam_size, device=src_ids.device)
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src_ids.device)

        context, _ = self.attention(decoder_hidden[-1], encoder_outputs, padding_mask)
        context = context.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
        decoder_hidden = decoder_hidden.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(-1, batch_size * beam_size, -1)

        for _ in range(max_length):
            input_tokens = sequences[:, :, -1].reshape(batch_size * beam_size, 1)
            logits, decoder_hidden = self.decoder(input_tokens, decoder_hidden, context)
            log_probs = torch.log_softmax(logits[:, -1], dim=-1)

            log_probs = log_probs.view(batch_size, beam_size, -1)
            log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
            cum_scores = scores.unsqueeze(-1) + log_probs

            top_scores, top_indices = cum_scores.view(batch_size, -1).topk(beam_size, dim=-1)
            beam_indices = top_indices // log_probs.size(-1)
            token_indices = top_indices % log_probs.size(-1)

            new_sequences = []
            for b in range(batch_size):
                new_seq = []
                for k in range(beam_size):
                    new_seq.append(sequences[b, beam_indices[b, k]].clone())
                new_sequences.append(torch.stack(new_seq))
            sequences = torch.stack(new_sequences)
            sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)

            scores = top_scores
            finished = torch.gather(finished, 1, beam_indices) | token_indices.eq(eos_id)

            context, _ = self.attention(decoder_hidden[-1], encoder_outputs.repeat_interleave(beam_size, dim=1), padding_mask.repeat_interleave(beam_size, dim=0))
            decoder_hidden = decoder_hidden.view(-1, batch_size, beam_size, decoder_hidden.size(-1))
            decoder_hidden = torch.gather(decoder_hidden, 2, beam_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, decoder_hidden.size(-1)))
            decoder_hidden = decoder_hidden.reshape(-1, batch_size * beam_size, -1)

            if finished.all():
                break

        best_indices = scores.argmax(dim=-1)
        result = []
        for b in range(batch_size):
            result.append(sequences[b, best_indices[b], 1:])
        return torch.cat([r.unsqueeze(0) for r in result], dim=0)


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

    @torch.no_grad()
    def generate_beam_search(self, src_ids, src_lengths, bos_id, eos_id, max_length=128, beam_size=5):
        del src_lengths
        memory, source_padding = self._memory(src_ids)
        batch_size = src_ids.size(0)

        sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        scores = torch.zeros(batch_size, beam_size, device=src_ids.device)
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src_ids.device)

        memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(batch_size * beam_size, memory.size(1), memory.size(2))
        source_padding = source_padding.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)

        for _ in range(max_length):
            input_tokens = sequences.reshape(batch_size * beam_size, -1)
            target_padding = input_tokens.eq(self.pad_id)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(input_tokens.size(1), device=src_ids.device)

            output = self.transformer.decoder(
                self._embed_target(input_tokens), memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding,
            )
            logits = self.output_projection(output[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            log_probs = log_probs.view(batch_size, beam_size, -1)
            log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
            cum_scores = scores.unsqueeze(-1) + log_probs

            top_scores, top_indices = cum_scores.view(batch_size, -1).topk(beam_size, dim=-1)
            beam_indices = top_indices // log_probs.size(-1)
            token_indices = top_indices % log_probs.size(-1)

            new_sequences = []
            for b in range(batch_size):
                new_seq = []
                for k in range(beam_size):
                    new_seq.append(sequences[b, beam_indices[b, k]].clone())
                new_sequences.append(torch.stack(new_seq))
            sequences = torch.stack(new_sequences)
            sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)

            scores = top_scores
            finished = torch.gather(finished, 1, beam_indices) | token_indices.eq(eos_id)

            if finished.all():
                break

        best_indices = scores.argmax(dim=-1)
        result = []
        for b in range(batch_size):
            result.append(sequences[b, best_indices[b], 1:])
        return torch.cat([r.unsqueeze(0) for r in result], dim=0)
