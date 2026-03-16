import pandas as pd
import numpy as np

# 生成模拟的GARCH+RESNET融合特征：1000行（样本数）×100列（特征数，和论文一致）
np.random.seed(123)  # 固定随机数，结果可复现
fusion_feat = np.random.randn(1000, 100)  # 100维特征，对应论文表4.3
pd.DataFrame(fusion_feat).to_csv('garch_resnet_feature.csv', index=False)

# 生成模拟的真实波动率标签：1000行×1列（模型要预测的目标值）
vol_label = np.random.randn(1000, 1)
pd.DataFrame(vol_label, columns=['volatility']).to_csv('volatility_label.csv', index=False)

print("文件生成成功！garch_resnet_feature.csv 和 volatility_label.csv 已创建")
