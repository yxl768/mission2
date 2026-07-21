"""Training entry point for IUPAC-to-SELFIES molecular representation learning."""

import argparse
import json
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from seq2seq_mol.data_utils import IUPAC_COLUMN, SELFIES_COLUMN, load_dataframe
from seq2seq_mol.models import GRUDecoder, GRUEncoder, Seq2SeqGRU, TransformerSeq2Seq
from seq2seq_mol.tokenizer import build_iupac_tokenizer, build_selfies_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, frame, src_tokenizer, tgt_tokenizer):
        self.iupac = frame[IUPAC_COLUMN].astype(str).tolist()
        self.selfies = frame[SELFIES_COLUMN].astype(str).tolist()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.iupac)

    def __getitem__(self, index):
        return {
            "src_ids": torch.tensor(self.src_tokenizer.encode(self.iupac[index]), dtype=torch.long),
            "tgt_ids": torch.tensor(self.tgt_tokenizer.encode(self.selfies[index]), dtype=torch.long),
        }


def collate_batch(batch, src_pad_id, tgt_pad_id):
    source = [item["src_ids"] for item in batch]
    target = [item["tgt_ids"] for item in batch]
    return {
        "src_ids": nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=src_pad_id),
        "tgt_ids": nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=tgt_pad_id),
        "src_lengths": torch.tensor([len(sequence) for sequence in source], dtype=torch.long),
    }


def make_loader(dataset, batch_size, shuffle, src_pad_id, tgt_pad_id):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_batch, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    for batch in loader:
        source = batch["src_ids"].to(device)
        target = batch["tgt_ids"].to(device)
        lengths = batch["src_lengths"].to(device)
        with torch.set_grad_enabled(is_training):
            logits = model(source, target, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * source.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    chunks = []
    for batch in loader:
        chunks.append(
            model.encode(batch["src_ids"].to(device), batch["src_lengths"].to(device)).cpu().numpy()
        )
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _compute_metrics(generated, source_cpu, target_cpu, src_tokenizer, tgt_tokenizer, samples, prefix):
    correct_tokens = total_tokens = exact = examples = 0
    for source_ids, target_ids, output_ids in zip(source_cpu, target_cpu, generated.cpu()):
        reference = [x for x in target_ids.tolist()[1:] if x not in {tgt_tokenizer.pad_id, tgt_tokenizer.eos_id}]
        prediction = []
        for token in output_ids.tolist():
            if token == tgt_tokenizer.eos_id:
                break
            if token != tgt_tokenizer.pad_id:
                prediction.append(token)
        total_tokens += len(reference)
        correct_tokens += sum(a == b for a, b in zip(reference, prediction))
        exact += int(reference == prediction)
        examples += 1
        if len(samples) < 5:
            samples.append({
                "iupac": src_tokenizer.decode(source_ids.tolist()),
                "reference_selfies": tgt_tokenizer.decode(reference),
                f"predicted_selfies_{prefix}": tgt_tokenizer.decode(prediction),
            })
    return correct_tokens, total_tokens, exact, examples


@torch.no_grad()
def translation_metrics(model, loader, src_tokenizer, tgt_tokenizer, device, max_decode_length, beam_size=0):
    """Report greedy and beam-search exact-match rate and token accuracy."""
    model.eval()
    samples = []
    
    greedy_correct = greedy_total = greedy_exact = greedy_examples = 0
    for batch in loader:
        source = batch["src_ids"].to(device)
        target = batch["tgt_ids"].to(device)
        lengths = batch["src_lengths"].to(device)
        generated = model.generate(source, lengths, tgt_tokenizer.bos_id, tgt_tokenizer.eos_id, max_decode_length)
        c, t, e, ex = _compute_metrics(generated, source.cpu(), target.cpu(), src_tokenizer, tgt_tokenizer, samples, "greedy")
        greedy_correct += c
        greedy_total += t
        greedy_exact += e
        greedy_examples += ex

    result = {
        "greedy_token_accuracy": greedy_correct / max(greedy_total, 1),
        "greedy_exact_match": greedy_exact / max(greedy_examples, 1),
        "validation_examples": greedy_examples,
        "samples": samples,
    }

    if beam_size > 0:
        beam_correct = beam_total = beam_exact = beam_examples = 0
        for batch in loader:
            source = batch["src_ids"].to(device)
            target = batch["tgt_ids"].to(device)
            lengths = batch["src_lengths"].to(device)
            generated = model.generate_beam_search(source, lengths, tgt_tokenizer.bos_id, tgt_tokenizer.eos_id, max_decode_length, beam_size)
            c, t, e, ex = _compute_metrics(generated, source.cpu(), target.cpu(), src_tokenizer, tgt_tokenizer, samples, "beam")
            beam_correct += c
            beam_total += t
            beam_exact += e
            beam_examples += ex
        result.update({
            "beam_token_accuracy": beam_correct / max(beam_total, 1),
            "beam_exact_match": beam_exact / max(beam_examples, 1),
            "beam_size": beam_size,
        })

    return result


def build_model(args, src_vocab_size, tgt_vocab_size, src_pad_id):
    if args.model_type == "gru":
        encoder = GRUEncoder(src_vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout, bidirectional=True)
        decoder = GRUDecoder(tgt_vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout)
        return Seq2SeqGRU(encoder, decoder, src_pad_id)
    if args.embed_size % args.num_heads:
        raise ValueError("--embed-size must be divisible by --num-heads for a Transformer.")
    return TransformerSeq2Seq(
        src_vocab_size, tgt_vocab_size, args.embed_size, args.num_heads,
        args.num_layers, args.num_layers, args.hidden_size * 4, args.dropout, src_pad_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train an IUPAC-to-SELFIES Seq2Seq model")
    parser.add_argument("--data", required=True, help="Processed CSV produced by build_dataset.py")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--epochs", type=int, default=30, help="减少epochs缩短训练时间")
    parser.add_argument("--batch-size", type=int, default=64, help="增大batch_size提高训练效率")
    parser.add_argument("--embed-size", type=int, default=64, help="减小嵌入维度")
    parser.add_argument("--hidden-size", type=int, default=64, help="减小隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=1, help="减少网络层数")
    parser.add_argument("--num-heads", type=int, default=2, help="减少Transformer头数")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-decode-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning rate decay factor")
    parser.add_argument("--eval-interval", type=int, default=1, help="每N个epoch验证一次")
    parser.add_argument("--beam-size", type=int, default=0, help="beam search大小，0为不使用")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = load_dataframe(args.data).dropna(subset=[IUPAC_COLUMN, SELFIES_COLUMN]).reset_index(drop=True)
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    if len(frame) < 10:
        raise ValueError("At least 10 valid IUPAC/SELFIES pairs are required.")
    os.makedirs(args.output, exist_ok=True)

    indices = np.random.default_rng(args.seed).permutation(len(frame))
    validation_size = max(1, round(len(frame) * args.validation_fraction))
    validation_indices, train_indices = indices[:validation_size], indices[validation_size:]
    train_frame = frame.iloc[train_indices]
    src_tokenizer = build_iupac_tokenizer(train_frame[IUPAC_COLUMN])
    tgt_tokenizer = build_selfies_tokenizer(train_frame[SELFIES_COLUMN])
    dataset = TranslationDataset(frame, src_tokenizer, tgt_tokenizer)
    train_loader = make_loader(Subset(dataset, train_indices.tolist()), args.batch_size, True, src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    validation_loader = make_loader(Subset(dataset, validation_indices.tolist()), args.batch_size, False, src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    all_loader = make_loader(dataset, args.batch_size, False, src_tokenizer.pad_id, tgt_tokenizer.pad_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, len(src_tokenizer.vocab), len(tgt_tokenizer.vocab), src_tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.patience // 2, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id)
    history, best_loss = [], float("inf")
    checkpoint_path = os.path.join(args.output, "best_model.pt")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(), "model_type": args.model_type,
                "model_args": vars(args), "src_vocab": src_tokenizer.vocab, "tgt_vocab": tgt_tokenizer.vocab,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs with patience {args.patience}")
                break

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    embeddings = extract_embeddings(model, all_loader, device)
    np.save(os.path.join(args.output, "encoder_embeddings.npy"), embeddings)
    np.savez(os.path.join(args.output, "split_indices.npz"), train=train_indices, validation=validation_indices)
    pd.DataFrame(history).to_csv(os.path.join(args.output, "training_log.csv"), index=False)
    metrics = translation_metrics(model, validation_loader, src_tokenizer, tgt_tokenizer, device, args.max_decode_length)
    metrics["best_validation_loss"] = best_loss
    with open(os.path.join(args.output, "translation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"Saved best model, {len(embeddings)} encoder embeddings, and validation translation metrics to {args.output}")


if __name__ == "__main__":
    main()
