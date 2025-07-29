import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class ExerciseClassifier(nn.Module):
    """运动分类模型"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

class LSTMExerciseClassifier(nn.Module):
    """基于LSTM的运动分类模型"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        return self.classifier(pooled)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config_path: str = "./config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集"""
        
        dataset_config = self.config['training']['datasets'][dataset_name]
        data_path = Path(dataset_config['path'])
        
        features = np.load(data_path / "features.npy")
        labels = np.load(data_path / "labels.npy")
        
        # 验证并修正标签
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[l] for l in labels])
        
        logger.info(f"加载数据集 {dataset_name}: {features.shape[0]} 样本, {features.shape[1]} 特征")
        return features, labels
    
    def prepare_data_loaders(self, features: np.ndarray, labels: np.ndarray, 
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备数据加载器"""
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        # 创建数据集
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        # 划分训练/验证/测试集
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, num_epochs: int = 100) -> Dict:
        """训练模型"""
        
        model = model.to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler(enabled=self.device.type == 'cuda')
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast(device_type=self.device.type):
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # 学习率调度
            scheduler.step()
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 确保 models 目录存在
                Path('./models').mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), './models/best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"早停在第 {epoch+1} 轮，最佳验证准确率: {best_val_acc:.2f}%")
                break
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      class_names: Optional[List[str]] = None) -> Dict:
        """评估模型"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # 计算准确率
        accuracy = 100.0 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # 生成分类报告
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(max(all_labels) + 1)]
        
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def plot_training_history(self, history: Dict, save_path: str = "./models/training_history.png"):
        """绘制训练历史"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 准确率曲线
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"训练历史图保存到: {save_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: str = "./models/confusion_matrix.png"):
        """绘制混淆矩阵"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"混淆矩阵保存到: {save_path}")

def main():
    """主训练流程"""
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 加载合成健身数据集
    features, labels = trainer.load_dataset('uci_har')
    logger.info(f"Labels range: min={np.min(labels)}, max={np.max(labels)}")
    unique_labels = np.unique(labels)
    logger.info(f"Unique labels: {unique_labels}, Number of classes: {len(unique_labels)}")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = trainer.prepare_data_loaders(features, labels)
    
    # 创建模型
    num_classes = len(unique_labels)
    input_dim = features.shape[1]
    
    model = ExerciseClassifier(input_dim=input_dim, num_classes=num_classes)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    history = trainer.train_model(model, train_loader, val_loader, num_epochs=100)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('./models/best_model.pth'))
    
    # 评估模型
    class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    results = trainer.evaluate_model(model, test_loader, class_names)
    
    logger.info(f"测试准确率: {results['accuracy']:.2f}%")
    
    # 绘制结果
    trainer.plot_training_history(history)
    trainer.plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # 转换为TorchScript进行部署
    model.eval()
    device = next(model.parameters()).device
    example_input = torch.randn(1, input_dim, device=device)
    traced_model = torch.jit.trace(model, example_input)
    # 确保 models 目录存在
    Path('./models').mkdir(parents=True, exist_ok=True)
    traced_model.save('./models/exercise_classifier.pt')
    
    logger.info("✓ 模型训练完成并保存为TorchScript格式")

if __name__ == "__main__":
    main()