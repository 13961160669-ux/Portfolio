import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def set_seed(seed=42, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到：{file_path}")


def check_file_format(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError(f"文件格式错误：{file_path} 必须是CSV格式")
    try:
        pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f" CSV文件读取失败：{file_path}，错误信息：{str(e)}")


def check_data_consistency(data, label):
    if len(data) != len(label):
        raise ValueError(f"数据长度不匹配：特征{len(data)}条，标签{len(label)}条")


def check_feature_dim(data, expected_dim=100):
    if data.shape[1] != expected_dim:
        raise ValueError(f" 特征维度错误：当前{data.shape[1]}维，要求100维")


class GRU_Volatility(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(GRU_Volatility, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        gru_out = self.activation(gru_out)
        out = self.fc(gru_out[:, -1, :])
        return out



set_seed(seed=42, deterministic=True)


input_size = 100
hidden_size = 100
num_layers = 2
dropout_rate = 0.4
model = GRU_Volatility(input_size, hidden_size, num_layers, dropout_rate)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
huber_loss = nn.HuberLoss(delta=1.35)


new_folder_path = r"C:\Users\89898\Desktop\新建文件夹 (2)\03_gru_replication"
data_path = os.path.join(new_folder_path, "garch_resnet_feature.csv")
label_path = os.path.join(new_folder_path, "volatility_label.csv")


check_file_exists(data_path)
check_file_exists(label_path)
check_file_format(data_path)
check_file_format(label_path)

data = pd.read_csv(data_path)
label_df = pd.read_csv(label_path)

data = data.loc[:, ~data.columns.str.contains('Unnamed')]
label_df = label_df.loc[:, ~label_df.columns.str.contains('Unnamed')]


assert label_df.shape[1] == 1, "标签文件只能有1列波动率标签！"
label = label_df.iloc[:, :1] 

check_data_consistency(data, label)
check_feature_dim(data, expected_dim=100)



def create_seq_data(data, label, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, :])
        y.append(label[i, :])
    return np.array(X), np.array(y)


seq_len = 10
X_all, y_all = create_seq_data(data.values, label.values, seq_len)


train_size = int(0.8 * len(X_all))
X_train_raw, X_test_raw = X_all[:train_size], X_all[train_size:]
y_train_raw, y_test_raw = y_all[:train_size], y_all[train_size:]


scaler_feat = StandardScaler()
scaler_label = StandardScaler()

X_train_scaled = scaler_feat.fit_transform(
    X_train_raw.reshape(-1, X_train_raw.shape[2])
).reshape(X_train_raw.shape)

X_test_scaled = scaler_feat.transform(
    X_test_raw.reshape(-1, X_test_raw.shape[2])
).reshape(X_test_raw.shape)

y_train_scaled = scaler_label.fit_transform(y_train_raw)
y_test_scaled = scaler_label.transform(y_test_raw)


X_train = torch.FloatTensor(X_train_scaled).to(device)
y_train = torch.FloatTensor(y_train_scaled).to(device)
X_test = torch.FloatTensor(X_test_scaled).to(device)
y_test = torch.FloatTensor(y_test_scaled).to(device)


batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


epochs = 100
train_loss_list = []
max_grad_norm = 1.0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_train_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}')


model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test)


y_pred = scaler_label.inverse_transform(y_pred_scaled.cpu().numpy())
y_true = scaler_label.inverse_transform(y_test.cpu().numpy())


mae_original = mean_absolute_error(y_true, y_pred)
mse_original = mean_squared_error(y_true, y_pred)


mse_scaled = criterion(y_pred_scaled, y_test).item()
huber_scaled = huber_loss(y_pred_scaled, y_test).item()

print('=' * 70)
print(f'GRU 波动率预测 - 最终评估')
print('=' * 70)
print(f'【原始尺度】  MAE: {mae_original:.6f}')
print(f'【原始尺度】  MSE: {mse_original:.6f}')
print(f'【标准化尺度】 MSE: {mse_scaled:.6f}')
print(f'【标准化尺度】 Huber: {huber_scaled:.6f}')
print('=' * 70)
print("说明：")
print("  - MAE/MSE(原始)：符合金融波动率预测可解释性")
print("  - MSE/Huber(标准化)：与训练损失同分布，delta参数有效")
print('=' * 70)


torch.save(model.state_dict(), 'gru_volatility_model.pth')
print("模型已保存：gru_volatility_model.pth")

# ---------------------- 融合预留 ----------------------
# garch_pred = pd.read_csv('garch_volatility_pred.csv')
# resnet_feat = pd.read_csv('resnet_feature.csv')
# fusion_feat = pd.concat([garch_pred, resnet_feat], axis=1)
# fusion_feat.to_csv('garch_resnet_feature.csv', index=False)
