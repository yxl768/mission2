import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from data_preprocessing import load_and_preprocess_data
from seq2seq_gru import GRUSeq2Seq
from seq2seq_transformer import TransformerSeq2Seq
from property_prediction import PropertyPredictor

# 配置参数解析器
def get_args():
    parser = argparse.ArgumentParser(description='分子表示学习与性质预测训练脚本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/zinc250k.csv', 
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--split_ratio', type=float, default=0.8, 
                        help='训练验证分割比例')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='transformer', choices=['gru', 'transformer'],
                        help='Seq2Seq模型类型: gru 或 transformer')
    parser.add_argument('--embed_dim', type=int, default=256, 
                        help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512, 
                        help='隐藏层维度(GRU专用)')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='编码器/解码器层数')
    parser.add_argument('--nhead', type=int, default=8, 
                        help='注意力头数(Transformer专用)')
    parser.add_argument('--dim_feedforward', type=int, default=1024, 
                        help='前馈网络维度(Transformer专用)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, 
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, 
                        help='教师强制比例')
    
    # 性质预测参数
    parser.add_argument('--property_hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='性质预测模型隐藏层维度')
    parser.add_argument('--property_dropout', type=float, default=0.2, 
                        help='性质预测模型dropout率')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='输出目录')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='模型保存间隔(轮数)')
    
    return parser.parse_args()

# 训练Seq2Seq模型
def train_seq2seq(model, train_loader, val_loader, args, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding索引
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        for src, tgt in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} - Train'):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            output = model(src, tgt)
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} - Val'):
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_{args.model_type}_seq2seq.pth'))
        
        # 定期保存模型
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model_type}_seq2seq_epoch{epoch}.pth'))
    
    return train_losses, val_losses

# 训练性质预测模型
def train_property_predictor(model, train_loader, val_loader, args, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 多任务损失函数
    criterion = []
    for task_type in args.property_task_types:
        if task_type == 'regression':
            criterion.append(nn.MSELoss())
        else:
            criterion.append(nn.CrossEntropyLoss())
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, args.property_epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        for reps, properties in tqdm(train_loader, desc=f'Epoch {epoch}/{args.property_epochs} - Property Train'):
            reps = reps.to(device)
            properties = [prop.to(device) for prop in properties]
            
            optimizer.zero_grad()
            outputs = model(reps)
            
            # 计算多任务损失
            loss = 0
            for i, (output, prop) in enumerate(zip(outputs, properties)):
                if args.property_task_types[i] == 'regression':
                    loss += criterion[i](output, prop)
                else:
                    loss += criterion[i](output, prop.long())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for reps, properties in tqdm(val_loader, desc=f'Epoch {epoch}/{args.property_epochs} - Property Val'):
                reps = reps.to(device)
                properties = [prop.to(device) for prop in properties]
                outputs = model(reps)
                
                loss = 0
                for i, (output, prop) in enumerate(zip(outputs, properties)):
                    if args.property_task_types[i] == 'regression':
                        loss += criterion[i](output, prop)
                    else:
                        loss += criterion[i](output, prop.long())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Property Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_property_predictor.pth'))
    
    return train_losses, val_losses

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('='*50)
    print('开始数据加载与预处理...')
    # 加载并预处理数据
    (train_src, train_tgt, val_src, val_tgt, 
     src_vocab_size, tgt_vocab_size, 
     property_train_reps, property_val_reps, 
     property_train_labels, property_val_labels,
     args.property_task_types) = load_and_preprocess_data(args.data_path, args.split_ratio)
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_src, train_tgt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_src, val_tgt)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print('='*50)
    print(f'开始训练 {args.model_type.upper()} Seq2Seq 模型...')
    # 初始化Seq2Seq模型
    if args.model_type == 'gru':
        model = GRUSeq2Seq(src_vocab_size, tgt_vocab_size, args.embed_dim, 
                           args.hidden_dim, args.num_layers)
    else:
        model = TransformerSeq2Seq(src_vocab_size, tgt_vocab_size, args.embed_dim, 
                                   args.nhead, args.num_layers, args.num_layers, 
                                   args.dim_feedforward)
    
    # 训练Seq2Seq模型
    seq2seq_train_loss, seq2seq_val_loss = train_seq2seq(
        model, train_loader, val_loader, args, device
    )
    
    print('='*50)
    print('提取分子表示...')
    # 提取分子表示
    model.eval()
    with torch.no_grad():
        # 提取训练集分子表示
        train_reps = []
        for src, _ in tqdm(train_loader, desc='提取训练集分子表示'):
            reps = model.get_molecular_representation(src.to(device))
            train_reps.append(reps.cpu())
        property_train_reps = torch.cat(train_reps, dim=0)
        
        # 提取验证集分子表示
        val_reps = []
        for src, _ in tqdm(val_loader, desc='提取验证集分子表示'):
            reps = model.get_molecular_representation(src.to(device))
            val_reps.append(reps.cpu())
        property_val_reps = torch.cat(val_reps, dim=0)
    
    print('='*50)
    print('开始训练性质预测模型...')
    # 创建性质预测数据加载器
    property_train_dataset = TensorDataset(property_train_reps, *property_train_labels)
    property_train_loader = DataLoader(property_train_dataset, batch_size=args.batch_size, shuffle=True)
    
    property_val_dataset = TensorDataset(property_val_reps, *property_val_labels)
    property_val_loader = DataLoader(property_val_dataset, batch_size=args.batch_size)
    
    # 初始化性质预测模型
    property_output_dims = [labels.shape[1] for labels in property_train_labels]
    property_model = PropertyPredictor(
        input_dim=args.embed_dim,
        hidden_dims=args.property_hidden_dims,
        output_dims=property_output_dims,
        task_types=args.property_task_types,
        dropout=args.property_dropout
    )
    
    # 训练性质预测模型
    property_train_loss, property_val_loss = train_property_predictor(
        property_model, property_train_loader, property_val_loader, args, device
    )
    
    print('='*50)
    print('训练完成!')
    print(f'Seq2Seq模型保存在: {args.output_dir}')
    print(f'性质预测模型保存在: {args.output_dir}/best_property_predictor.pth')

if __name__ == "__main__":
    main()