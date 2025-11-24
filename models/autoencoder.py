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