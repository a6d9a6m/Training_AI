import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class AudioAutoencoder(nn.Module):
    """
    基于深度学习的音频自动编码器模型，用于无监督异常检测
    通过重构误差来识别异常样本
    """
    
    def __init__(self, input_dim=1024, encoding_dim=128, dropout_rate=0.3):
        """
        初始化自动编码器模型
        
        参数:
        input_dim: 输入特征维度
        encoding_dim: 编码器输出维度（潜在空间维度）
        dropout_rate: Dropout率，用于防止过拟合
        """
        super(AudioAutoencoder, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, input_dim)
            # 输出层不使用激活函数，因为我们希望能够重构原始特征范围
        )
        
        # 特征标准化器
        self.scaler = None
        
        # 重构误差阈值，用于后续推理
        self.reconstruction_threshold = None
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 输入特征张量
        
        返回:
        encoded: 编码后的潜在特征
        decoded: 解码后的重构特征
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """
        仅编码部分，用于获取特征表示
        
        参数:
        x: 输入特征张量
        
        返回:
        encoded: 编码后的潜在特征
        """
        return self.encoder(x)
    
    def decode(self, encoded):
        """
        仅解码部分，用于从潜在特征重构
        
        参数:
        encoded: 编码后的潜在特征
        
        返回:
        decoded: 解码后的重构特征
        """
        return self.decoder(encoded)
    
    def train_model(self, train_loader, optimizer, criterion, epochs=100, verbose=True):
        """
        训练自动编码器模型
        
        参数:
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epochs: 训练轮数
        verbose: 是否打印训练进度
        
        返回:
        train_losses: 每轮训练的损失
        """
        train_losses = []
        
        self.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_features, _ in train_loader:  # 忽略标签，因为是无监督学习
                # 移动到指定设备
                batch_features = batch_features.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                _, outputs = self(batch_features)
                
                # 计算损失
                loss = criterion(outputs, batch_features)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_features.size(0)
            
            # 计算平均损失
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}')
        
        if verbose:
            print('训练完成')
        
        return train_losses
    
    def fit_scaler(self, train_data):
        """
        拟合特征标准化器
        
        参数:
        train_data: 用于拟合标准化器的训练数据
        """
        # 收集所有训练数据特征
        all_features = []
        for features, _ in train_data:
            # 确保features是2D数组 (样本数, 特征数)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            all_features.append(features)
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        
        # 拟合标准化器
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)
    
    def transform_features(self, features):
        """
        使用拟合好的标准化器转换特征
        
        参数:
        features: 要转换的特征
        
        返回:
        transformed: 转换后的特征
        """
        if self.scaler is None:
            raise ValueError("标准化器未拟合，请先调用fit_scaler方法")
        
        # 确保features是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 转换特征
        transformed = self.scaler.transform(features)
        return transformed
    
    def set_reconstruction_threshold(self, normal_data_loader, percentile=95):
        """
        设置重构误差阈值
        
        参数:
        normal_data_loader: 只包含正常样本的数据加载器
        percentile: 用于计算阈值的百分位数
        """
        self.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_features, _ in normal_data_loader:
                batch_features = batch_features.to(self.device)
                _, outputs = self(batch_features)
                
                # 计算每个样本的重构误差 (MSE)
                batch_errors = torch.mean((outputs - batch_features) ** 2, dim=1)
                reconstruction_errors.extend(batch_errors.cpu().numpy())
        
        # 设置阈值
        self.reconstruction_threshold = np.percentile(reconstruction_errors, percentile)
        print(f"重构误差阈值设置为: {self.reconstruction_threshold:.6f} (基于正常样本的{percentile}%分位数)")
    
    def predict_anomaly(self, features, return_scores=False):
        """
        预测样本是否异常
        
        参数:
        features: 输入特征
        return_scores: 是否返回异常分数
        
        返回:
        当return_scores=False时返回预测标签 (0=正常, 1=异常)
        当return_scores=True时返回 (预测标签, 异常分数)
        """
        self.eval()
        
        # 确保输入是张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 移动到设备
        features = features.to(self.device)
        
        with torch.no_grad():
            _, outputs = self(features)
            
            # 计算重构误差 (MSE)
            reconstruction_error = torch.mean((outputs - features) ** 2, dim=1)
        
        # 判断是否异常
        if self.reconstruction_threshold is None:
            raise ValueError("重构误差阈值未设置，请先调用set_reconstruction_threshold方法")
        
        anomalies = (reconstruction_error > self.reconstruction_threshold).cpu().numpy()
        
        if return_scores:
            return anomalies.astype(int), reconstruction_error.cpu().numpy()
        else:
            return anomalies.astype(int)


def create_autoencoder_model(input_dim=1024, encoding_dim=128):
    """
    创建并返回自动编码器模型实例
    
    参数:
    input_dim: 输入特征维度
    encoding_dim: 编码器输出维度
    
    返回:
    model: 自动编码器模型实例
    """
    model = AudioAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    return model


def get_autoencoder_loss():
    """
    获取自动编码器的损失函数

    返回:
    criterion: 均方误差损失函数
    """
    return nn.MSELoss()


class AutoencoderModel:
    """
    自动编码器模型包装类 - 提供统一的训练和预测接口
    """

    def __init__(self, input_dim=1024, encoding_dim=128, dropout_rate=0.3):
        """
        初始化自动编码器模型

        参数:
        input_dim: 输入特征维度
        encoding_dim: 编码维度
        dropout_rate: Dropout率
        """
        self.model = AudioAutoencoder(input_dim, encoding_dim, dropout_rate)
        self.threshold = None

    def fit(self, X_train, epochs=100, batch_size=32, validation_data=None, learning_rate=0.001):
        """
        训练自动编码器

        参数:
        X_train: 训练数据 (numpy array)
        epochs: 训练轮数
        batch_size: 批次大小
        validation_data: 验证数据
        learning_rate: 学习率
        """
        # 转换为PyTorch数据集
        from torch.utils.data import TensorDataset, DataLoader

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_dummy = torch.zeros(len(X_train))  # 占位标签
        train_dataset = TensorDataset(X_tensor, y_dummy)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # 训练
        print("开始训练自动编码器...")
        self.model.train_model(train_loader, optimizer, criterion, epochs=epochs, verbose=True)

        # 计算阈值
        if validation_data is not None:
            X_val_tensor = torch.tensor(validation_data, dtype=torch.float32)
            y_val_dummy = torch.zeros(len(validation_data))
            val_dataset = TensorDataset(X_val_tensor, y_val_dummy)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.model.set_reconstruction_threshold(val_loader, percentile=95)
            self.threshold = self.model.reconstruction_threshold

    def predict(self, X):
        """
        预测重构

        参数:
        X: 输入数据

        返回:
        重构后的数据
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.model.device)

        with torch.no_grad():
            _, reconstructed = self.model(X_tensor)

        return reconstructed.cpu().numpy()

    def predict_anomaly(self, X, threshold=None):
        """
        预测异常

        参数:
        X: 输入数据
        threshold: 阈值（如果为None，使用训练时计算的阈值）

        返回:
        predictions: 预测标签 (0=正常, 1=异常)
        """
        if threshold is not None:
            self.model.reconstruction_threshold = threshold
            self.threshold = threshold

        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = self.model.predict_anomaly(X_tensor)

        return predictions

    def save(self, path):
        """
        保存模型

        参数:
        path: 保存路径
        """
        import os
        os.makedirs(path, exist_ok=True)

        # 保存模型权重
        model_file = os.path.join(path, 'model.pth')
        torch.save(self.model.state_dict(), model_file)

        # 保存配置
        import pickle
        config_file = os.path.join(path, 'config.pkl')
        with open(config_file, 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'input_dim': list(self.model.encoder[0].parameters())[0].shape[1],
                'encoding_dim': self.model.encoding_dim
            }, f)

        print(f"模型已保存到: {path}")

    def load(self, path):
        """
        加载模型

        参数:
        path: 模型路径
        """
        import pickle

        # 加载配置
        config_file = os.path.join(path, 'config.pkl')
        with open(config_file, 'rb') as f:
            config = pickle.load(f)

        # 重新创建模型
        self.model = AudioAutoencoder(
            input_dim=config['input_dim'],
            encoding_dim=config['encoding_dim']
        )

        # 加载权重
        model_file = os.path.join(path, 'model.pth')
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

        # 恢复阈值
        self.threshold = config['threshold']
        self.model.reconstruction_threshold = self.threshold

        print(f"模型已从 {path} 加载")