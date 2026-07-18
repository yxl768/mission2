import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

class PropertyPredictor(nn.Module):
    """分子性质预测模型"""
    def __init__(self, input_dim, hidden_dims, output_dims, task_types, dropout=0.2):
        """
        初始化性质预测模型
        
        参数:
            input_dim: 输入维度 (分子表示的维度)
            hidden_dims: 隐藏层维度列表，如 [512, 256]
            output_dims: 每个任务的输出维度列表
            task_types: 每个任务的类型列表，'regression' 或 'classification'
            dropout: Dropout概率
        """
        super(PropertyPredictor, self).__init__()
        
        # 验证参数
        assert len(output_dims) == len(task_types), "输出维度和任务类型数量必须一致"
        self.num_tasks = len(output_dims)
        self.task_types = task_types
        
        # 创建共享的隐藏层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 创建任务特定的输出层
        self.task_heads = nn.ModuleList()
        for dim in output_dims:
            self.task_heads.append(nn.Linear(prev_dim, dim))
    
    def forward(self, x):
        """前向传播"""
        # 共享层
        shared_output = self.shared_layers(x)
        
        # 任务特定输出
        outputs = []
        for head in self.task_heads:
            outputs.append(head(shared_output))
        
        return outputs
    
    def predict(self, x):
        """预测模式"""
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = []
            
            for i, output in enumerate(outputs):
                if self.task_types[i] == 'classification':
                    # 分类任务: 应用softmax并取argmax
                    pred = torch.softmax(output, dim=1).argmax(dim=1)
                else:
                    # 回归任务: 直接使用输出
                    pred = output
                predictions.append(pred.cpu().numpy())
            
            return predictions
    
    def evaluate(self, x, y_true):
        """评估模型性能"""
        predictions = self.predict(x)
        results = {}
        
        for i in range(self.num_tasks):
            task_type = self.task_types[i]
            pred = predictions[i]
            true = y_true[i].cpu().numpy()
            
            if task_type == 'regression':
                # 回归任务指标
                mse = mean_squared_error(true, pred)
                r2 = r2_score(true, pred)
                results[f'task_{i}'] = {
                    'type': 'regression',
                    'mse': mse,
                    'r2': r2
                }
            else:
                # 分类任务指标
                acc = accuracy_score(true, pred)
                f1 = f1_score(true, pred, average='weighted')
                results[f'task_{i}'] = {
                    'type': 'classification',
                    'accuracy': acc,
                    'f1_score': f1
                }
        
        return results

# 示例用法
if __name__ == "__main__":
    # 模拟参数
    input_dim = 512  # 分子表示维度
    hidden_dims = [256, 128]  # 隐藏层
    output_dims = [1, 3]  # 任务1: 回归(输出1维), 任务2: 分类(3类)
    task_types = ['regression', 'classification']
    
    # 创建模型
    model = PropertyPredictor(input_dim, hidden_dims, output_dims, task_types)
    
    # 模拟输入
    batch_size = 32
    molecular_reps = torch.randn(batch_size, input_dim)
    
    # 前向传播
    outputs = model(molecular_reps)
    print(f"任务1输出形状: {outputs[0].shape}")  # (32, 1)
    print(f"任务2输出形状: {outputs[1].shape}")  # (32, 3)
    
    # 模拟真实标签
    y_true_reg = torch.randn(batch_size, 1)
    y_true_cls = torch.randint(0, 3, (batch_size,))
    
    # 评估模型
    y_true = [y_true_reg, y_true_cls]
    results = model.evaluate(molecular_reps, y_true)
    print("评估结果:", results)