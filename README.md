# PCDN流量分类项目

## 📋 项目概述

本项目使用XGBoost机器学习算法对网络流量进行二分类，区分正常流量和PCDN流量。项目提供完整的数据处理、模型训练、评估和可视化功能。

### 🎯 分类目标
- **APP_0**: 正常流量 (标签: 0)  
- **APP_1**: PCDN流量 (标签: 1)

## 📁 项目结构

```
temp_xgboost/
├── 📓 PCDN_Traffic_Classification_XGBoost.ipynb  # 主要分析notebook
├── 📄 README.md                                 # 项目说明文档
└── 📁 pcdn_32_pkts_2class_feature_enhance_v17.4_dataset/
    ├── Training_set/                            # 训练数据
    │   ├── APP_0/ (正常流量)
    │   └── APP_1/ (PCDN流量)
    ├── Validation_set/                          # 验证数据
    │   ├── APP_0/
    │   └── APP_1/
    └── Testing_set/                             # 测试数据
        ├── APP_0/
        └── APP_1/
```

## 🔧 环境要求

### Python版本
- Python 3.7+

### 必需依赖包
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 可选依赖（中文字体支持）
```bash
# Windows用户通常已包含SimHei字体
# Linux/Mac用户可能需要安装中文字体
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆或下载项目到本地
cd temp_xgboost

# 安装依赖包
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 2. 启动Jupyter Notebook
```bash
jupyter notebook
```

### 3. 运行分析
1. 在浏览器中打开 `PCDN_Traffic_Classification_XGBoost.ipynb`
2. 依次运行所有单元格（Cell -> Run All）
3. 等待训练完成，查看结果

## 📊 功能特性

### 🔍 数据分析
- ✅ 自动加载训练/验证/测试数据集
- ✅ 数据质量检查和统计分析
- ✅ 缺失值处理和数据清洗

### 🛠️ 特征工程
- ✅ 网络流量特征提取（32包序列）
- ✅ 统计特征生成（均值、标准差、最大值等）
- ✅ 分类特征编码
- ✅ 载荷数据长度特征

### 🤖 模型训练
- ✅ XGBoost二分类器
- ✅ 超参数优化配置
- ✅ 交叉验证支持
- ✅ 实时训练监控

### 📈 模型评估
- ✅ 准确率、精确率、召回率、F1分数
- ✅ ROC曲线和AUC指标
- ✅ 混淆矩阵热图
- ✅ 预测概率分布分析

### 🔬 特征分析
- ✅ 特征重要性排序
- ✅ Top N重要特征可视化
- ✅ 累积重要性贡献分析
- ✅ 特征选择建议

### 💾 结果输出
- ✅ 训练好的模型文件 (.pkl)
- ✅ 特征配置文件 (.json)
- ✅ 特征重要性报告 (.csv)

## 📋 详细使用步骤

### Step 1: 数据加载
notebook会自动：
- 扫描数据集目录
- 加载所有CSV文件
- 合并同类型数据
- 添加标签信息

### Step 2: 数据预处理
自动处理：
- 缺失值填充
- 特殊字段解析（ip_direction, pkt_len, iat, payload）
- 类别特征编码
- 数值特征标准化

### Step 3: 模型训练
XGBoost配置：
```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### Step 4: 结果分析
生成多种可视化图表：
- 🔥 混淆矩阵
- 📈 ROC曲线
- 🎯 预测概率分布
- 🏆 特征重要性排序
- 📊 累积重要性贡献

## 🎨 可视化示例

项目提供丰富的可视化分析：

1. **数据分布分析**
   - 标签分布饼图
   - 数值特征直方图
   - 相关性热图

2. **模型性能评估**
   - 训练/验证/测试准确率对比
   - ROC曲线和AUC值
   - 混淆矩阵热图

3. **特征重要性分析**
   - Top 20特征重要性条形图
   - 重要性分布直方图
   - 累积贡献度曲线
   - Top 10特征占比饼图

## 📝 输出文件说明

运行完成后会生成以下文件：

### 1. pcdn_traffic_classifier.pkl
- 训练好的XGBoost模型
- 可直接用于新数据预测

### 2. model_features.json
- 模型使用的特征列表
- 确保新数据预处理一致性

### 3. feature_importance.csv
- 详细的特征重要性分析
- 包含特征名称和重要性分数

## 🔄 模型使用示例

```python
import joblib
import json
import pandas as pd

# 加载模型和特征配置
model = joblib.load('pcdn_traffic_classifier.pkl')
with open('model_features.json', 'r') as f:
    feature_columns = json.load(f)

# 预处理新数据（需要与训练数据相同的预处理步骤）
# new_data = preprocess_new_data(raw_data)

# 进行预测
# predictions = model.predict(new_data[feature_columns])
# probabilities = model.predict_proba(new_data[feature_columns])
```

## ⚙️ 超参数调优建议

如需进一步优化模型性能，可调整以下参数：

```python
# 深度和复杂度
'max_depth': [3, 6, 9],           # 树的最大深度
'min_child_weight': [1, 3, 5],   # 叶子节点最小样本权重

# 学习率和迭代次数
'learning_rate': [0.01, 0.1, 0.2],  # 学习率
'n_estimators': [100, 200, 500],    # 树的数量

# 正则化
'reg_alpha': [0, 0.1, 1],        # L1正则化
'reg_lambda': [1, 1.5, 2],       # L2正则化

# 采样
'subsample': [0.8, 0.9, 1.0],    # 行采样比例
'colsample_bytree': [0.8, 0.9, 1.0]  # 列采样比例
```

## 🐛 常见问题

### Q1: 报错"ModuleNotFoundError"
**解决方案**: 安装缺失的依赖包
```bash
pip install [包名]
```

### Q2: 中文字体显示问题
**解决方案**: 检查系统字体或修改字体配置
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
```

### Q3: 内存不足
**解决方案**: 
- 减少数据量或批量处理
- 降低XGBoost参数复杂度
- 增加虚拟内存

### Q4: 训练时间过长
**解决方案**:
- 减少n_estimators数量
- 增加learning_rate
- 使用GPU加速（需要xgboost-gpu）

## 📚 参考资料

- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- [Scikit-learn用户指南](https://scikit-learn.org/stable/user_guide.html)
- [Pandas数据处理](https://pandas.pydata.org/docs/)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。

---

🎉 **祝您使用愉快！如有问题，请查看notebook中的详细输出信息。**
