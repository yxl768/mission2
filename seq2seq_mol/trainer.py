"""IUPAC到SELFIES分子表示学习的训练入口."""

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
    """翻译数据集 - 处理IUPAC名称到SELFIES的配对数据.
    
    将数据集中的IUPAC名称和SELFIES转换为token ID序列，供模型训练使用。
    """

    def __init__(self, frame, src_tokenizer, tgt_tokenizer):
        """初始化翻译数据集.
        
        Args:
            frame: 包含IUPAC和SELFIES列的DataFrame
            src_tokenizer: 源序列（IUPAC）的tokenizer
            tgt_tokenizer: 目标序列（SELFIES）的tokenizer
        """
        self.iupac = frame[IUPAC_COLUMN].astype(str).tolist()
        self.selfies = frame[SELFIES_COLUMN].astype(str).tolist()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        """返回数据集大小."""
        return len(self.iupac)

    def __getitem__(self, index):
        """获取单个样本.
        
        Args:
            index: 样本索引
        
        Returns:
            dict: 包含src_ids（IUPAC编码）和tgt_ids（SELFIES编码）
        """
        return {
            "src_ids": torch.tensor(self.src_tokenizer.encode(self.iupac[index]), dtype=torch.long),
            "tgt_ids": torch.tensor(self.tgt_tokenizer.encode(self.selfies[index]), dtype=torch.long),
        }


def collate_batch(batch, src_pad_id, tgt_pad_id):
    """批量数据处理 - 将变长序列padding到相同长度.
    
    Args:
        batch: 一批样本，每个样本包含src_ids和tgt_ids
        src_pad_id: 源序列padding token的ID
        tgt_pad_id: 目标序列padding token的ID
    
    Returns:
        dict: 包含padding后的src_ids、tgt_ids和src_lengths
    """
    source = [item["src_ids"] for item in batch]
    target = [item["tgt_ids"] for item in batch]
    return {
        "src_ids": nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=src_pad_id),
        "tgt_ids": nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=tgt_pad_id),
        "src_lengths": torch.tensor([len(sequence) for sequence in source], dtype=torch.long),
    }


def make_loader(dataset, batch_size, shuffle, src_pad_id, tgt_pad_id):
    """创建数据加载器.
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        src_pad_id: 源序列padding token的ID
        tgt_pad_id: 目标序列padding token的ID
    
    Returns:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_batch, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """运行一个训练/验证epoch.
    
    Args:
        model: 模型实例
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备（CPU/GPU）
        optimizer: 优化器（训练时提供，验证时为None）
    
    Returns:
        float: 平均损失
    """
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    
    for batch in loader:
        # 将数据移到指定设备
        source = batch["src_ids"].to(device)
        target = batch["tgt_ids"].to(device)
        lengths = batch["src_lengths"].to(device)
        
        with torch.set_grad_enabled(is_training):
            # 模型前向传播
            logits = model(source, target, lengths)
            # 计算损失：忽略padding位置
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
        
        if is_training:
            # 梯度清零
            optimizer.zero_grad(set_to_none=True)
            # 反向传播
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
        
        total_loss += loss.item() * source.size(0)
    
    # 返回平均损失
    return total_loss / len(loader.dataset)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """提取编码器嵌入 - 将所有样本编码为固定长度向量.
    
    Args:
        model: 训练好的模型
        loader: 数据加载器
        device: 计算设备
    
    Returns:
        np.ndarray: 所有样本的嵌入矩阵，形状 (num_samples, embedding_dim)
    """
    model.eval()
    chunks = []
    for batch in loader:
        # 编码每个batch的样本
        chunks.append(
            model.encode(batch["src_ids"].to(device), batch["src_lengths"].to(device)).cpu().numpy()
        )
    # 拼接所有batch的嵌入
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _compute_metrics(generated, source_cpu, target_cpu, src_tokenizer, tgt_tokenizer, samples, prefix):
    """计算翻译指标 - token准确率和精确匹配率.
    
    Args:
        generated: 生成的序列，形状 (batch_size, gen_len)
        source_cpu: 源序列（CPU），形状 (batch_size, src_len)
        target_cpu: 目标序列（CPU），形状 (batch_size, tgt_len)
        src_tokenizer: 源序列tokenizer
        tgt_tokenizer: 目标序列tokenizer
        samples: 用于存储示例的列表
        prefix: 指标前缀（greedy或beam）
    
    Returns:
        tuple: (正确token数, 总token数, 精确匹配数, 总样本数)
    """
    correct_tokens = total_tokens = exact = examples = 0
    
    for source_ids, target_ids, output_ids in zip(source_cpu, target_cpu, generated.cpu()):
        # 提取参考序列（去掉bos和padding/eos）
        reference = [x for x in target_ids.tolist()[1:] if x not in {tgt_tokenizer.pad_id, tgt_tokenizer.eos_id}]
        # 提取预测序列（遇到eos停止）
        prediction = []
        for token in output_ids.tolist():
            if token == tgt_tokenizer.eos_id:
                break
            if token != tgt_tokenizer.pad_id:
                prediction.append(token)
        
        # 计算指标
        total_tokens += len(reference)
        correct_tokens += sum(a == b for a, b in zip(reference, prediction))
        exact += int(reference == prediction)
        examples += 1
        
        # 存储前5个示例用于可视化
        if len(samples) < 5:
            samples.append({
                "iupac": src_tokenizer.decode(source_ids.tolist()),
                "reference_selfies": tgt_tokenizer.decode(reference),
                f"predicted_selfies_{prefix}": tgt_tokenizer.decode(prediction),
            })
    
    return correct_tokens, total_tokens, exact, examples


@torch.no_grad()
def translation_metrics(model, loader, src_tokenizer, tgt_tokenizer, device, max_decode_length, beam_size=0):
    """计算翻译任务的评估指标.
    
    报告贪婪搜索和束搜索的token准确率和精确匹配率。
    
    Args:
        model: 训练好的模型
        loader: 验证/测试数据加载器
        src_tokenizer: 源序列tokenizer
        tgt_tokenizer: 目标序列tokenizer
        device: 计算设备
        max_decode_length: 最大解码长度
        beam_size: 束搜索宽度，0表示不使用束搜索
    
    Returns:
        dict: 包含各项评估指标和示例
    """
    model.eval()
    samples = []
    
    # 贪婪搜索评估
    greedy_correct = greedy_total = greedy_exact = greedy_examples = 0
    for batch in loader:
        source = batch["src_ids"].to(device)
        target = batch["tgt_ids"].to(device)
        lengths = batch["src_lengths"].to(device)
        # 贪婪搜索生成
        generated = model.generate(source, lengths, tgt_tokenizer.bos_id, tgt_tokenizer.eos_id, max_decode_length)
        c, t, e, ex = _compute_metrics(generated, source.cpu(), target.cpu(), src_tokenizer, tgt_tokenizer, samples, "greedy")
        greedy_correct += c
        greedy_total += t
        greedy_exact += e
        greedy_examples += ex

    # 构建结果字典
    result = {
        "greedy_token_accuracy": greedy_correct / max(greedy_total, 1),
        "greedy_exact_match": greedy_exact / max(greedy_examples, 1),
        "validation_examples": greedy_examples,
        "samples": samples,
    }

    # 束搜索评估（如果启用）
    if beam_size > 0:
        beam_correct = beam_total = beam_exact = beam_examples = 0
        for batch in loader:
            source = batch["src_ids"].to(device)
            target = batch["tgt_ids"].to(device)
            lengths = batch["src_lengths"].to(device)
            # 束搜索生成
            generated = model.generate_beam_search(source, lengths, tgt_tokenizer.bos_id, tgt_tokenizer.eos_id, max_decode_length, beam_size)
            c, t, e, ex = _compute_metrics(generated, source.cpu(), target.cpu(), src_tokenizer, tgt_tokenizer, samples, "beam")
            beam_correct += c
            beam_total += t
            beam_exact += e
            beam_examples += ex
        
        # 添加束搜索指标
        result.update({
            "beam_token_accuracy": beam_correct / max(beam_total, 1),
            "beam_exact_match": beam_exact / max(beam_examples, 1),
            "beam_size": beam_size,
        })

    return result


def build_model(args, src_vocab_size, tgt_vocab_size, src_pad_id):
    """构建模型 - 根据参数选择GRU或Transformer模型.
    
    Args:
        args: 命令行参数
        src_vocab_size: 源词汇表大小
        tgt_vocab_size: 目标词汇表大小
        src_pad_id: 源序列padding token的ID
    
    Returns:
        nn.Module: 构建好的模型
    """
    if args.model_type == "gru":
        # 构建GRU编码器和解码器
        encoder = GRUEncoder(src_vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout, bidirectional=True)
        decoder = GRUDecoder(tgt_vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout)
        return Seq2SeqGRU(encoder, decoder, src_pad_id)
    
    # 构建Transformer模型
    if args.embed_size % args.num_heads:
        raise ValueError("--embed-size必须能被--num-heads整除，以支持多头注意力.")
    
    return TransformerSeq2Seq(
        src_vocab_size, tgt_vocab_size, args.embed_size, args.num_heads,
        args.num_layers, args.num_layers, args.hidden_size * 4, args.dropout, src_pad_id,
    )


def parse_args():
    """解析命令行参数.
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="训练IUPAC到SELFIES的Seq2Seq模型")
    parser.add_argument("--data", required=True, help="build_dataset.py生成的处理后CSV文件")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--embed-size", type=int, default=64, help="词嵌入维度")
    parser.add_argument("--hidden-size", type=int, default=64, help="隐藏层维度（GRU）或Transformer前馈网络维度的1/4")
    parser.add_argument("--num-layers", type=int, default=1, help="网络层数")
    parser.add_argument("--num-heads", type=int, default=2, help="Transformer注意力头数")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--max-decode-length", type=int, default=128, help="最大解码长度")
    parser.add_argument("--max-samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--patience", type=int, default=8, help="早停耐心值")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="学习率衰减因子")
    parser.add_argument("--eval-interval", type=int, default=1, help="每N个epoch验证一次")
    parser.add_argument("--beam-size", type=int, default=0, help="束搜索大小，0为不使用")
    return parser.parse_args()


def main():
    """主函数 - 训练流程入口."""
    # 解析参数
    args = parse_args()
    
    # 参数验证
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction必须在0和1之间.")
    
    # 设置随机种子，保证实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载数据
    frame = load_dataframe(args.data).dropna(subset=[IUPAC_COLUMN, SELFIES_COLUMN]).reset_index(drop=True)
    # 限制最大样本数
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    # 数据量检查
    if len(frame) < 10:
        raise ValueError("至少需要10条有效IUPAC/SELFIES配对数据.")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 划分训练集和验证集
    indices = np.random.default_rng(args.seed).permutation(len(frame))
    validation_size = max(1, round(len(frame) * args.validation_fraction))
    validation_indices, train_indices = indices[:validation_size], indices[validation_size:]
    train_frame = frame.iloc[train_indices]
    
    # 构建tokenizer（仅使用训练集数据）
    src_tokenizer = build_iupac_tokenizer(train_frame[IUPAC_COLUMN])
    tgt_tokenizer = build_selfies_tokenizer(train_frame[SELFIES_COLUMN])
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(frame, src_tokenizer, tgt_tokenizer)
    train_loader = make_loader(Subset(dataset, train_indices.tolist()), args.batch_size, True, src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    validation_loader = make_loader(Subset(dataset, validation_indices.tolist()), args.batch_size, False, src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    all_loader = make_loader(dataset, args.batch_size, False, src_tokenizer.pad_id, tgt_tokenizer.pad_id)

    # 选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建模型并移到设备
    model = build_model(args, len(src_tokenizer.vocab), len(tgt_tokenizer.vocab), src_tokenizer.pad_id).to(device)
    
    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.patience // 2, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id)
    
    # 训练状态变量
    history, best_loss = [], float("inf")
    checkpoint_path = os.path.join(args.output, "best_model.pt")
    patience_counter = 0

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 训练epoch
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        # 验证epoch
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        # 更新学习率
        scheduler.step(validation_loss)
        # 记录训练历史
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        # 打印训练信息
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # 检查是否为最佳模型
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "model_state": model.state_dict(), "model_type": args.model_type,
                "model_args": vars(args), "src_vocab": src_tokenizer.vocab, "tgt_vocab": tgt_tokenizer.vocab,
            }, checkpoint_path)
        else:
            # 早停检查
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停：在第{epoch}个epoch后，验证损失连续{args.patience}次未改善")
                break

    # 加载最佳模型
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    
    # 提取所有样本的嵌入
    embeddings = extract_embeddings(model, all_loader, device)
    np.save(os.path.join(args.output, "encoder_embeddings.npy"), embeddings)
    
    # 保存训练/验证集划分索引
    np.savez(os.path.join(args.output, "split_indices.npz"), train=train_indices, validation=validation_indices)
    
    # 保存训练日志
    pd.DataFrame(history).to_csv(os.path.join(args.output, "training_log.csv"), index=False)
    
    # 计算翻译指标
    metrics = translation_metrics(model, validation_loader, src_tokenizer, tgt_tokenizer, device, args.max_decode_length)
    metrics["best_validation_loss"] = best_loss
    
    # 保存翻译指标
    with open(os.path.join(args.output, "translation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    
    print(f"已保存最佳模型、{len(embeddings)}个编码器嵌入和验证翻译指标到 {args.output}")


if __name__ == "__main__":
    main()
