# PCDN vs Normal Traffic Classification using Decision Tree
# 使用 ip.proto 和 tcp.srcport 特征进行决策树二分类

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
sns.set_palette("husl")

print("🌳 开始PCDN流量决策树分类任务")
print("📋 使用特征: ip.proto, tcp.srcport")
print("🎯 模型: Decision Tree Classifier")
print("=" * 60)

# 定义数据路径
base_path = "pcdn_32_pkts_2class_feature_enhance_v17.4_dataset"
train_path = os.path.join(base_path, "Training_set")
val_path = os.path.join(base_path, "Validation_set") 
test_path = os.path.join(base_path, "Testing_set")

# 选择的特征
selected_features = ['ip.proto', 'tcp.srcport']

def load_data_from_directory(directory_path, label):
    """加载指定目录下的所有CSV文件"""
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    dataframes = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['label'] = label  # 添加标签
            df['source_file'] = os.path.basename(file)  # 记录来源文件
            dataframes.append(df)
            print(f"✅ 加载文件: {file} (样本数: {len(df)})")
        except Exception as e:
            print(f"❌ 加载文件失败: {file}, 错误: {e}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"📊 {directory_path} 总样本数: {len(combined_df)}")
        return combined_df
    else:
        print(f"⚠️  {directory_path} 没有找到有效数据")
        return pd.DataFrame()

# 检查是否有空的数据集
def safe_concat(dataframes, set_name):
    """安全合并数据集，处理空数据集的情况"""
    non_empty_dfs = [df for df in dataframes if not df.empty]
    if not non_empty_dfs:
        print(f"⚠️  {set_name} 没有有效数据!")
        return pd.DataFrame()
    return pd.concat(non_empty_dfs, ignore_index=True)

def preprocess_features(data, features):
    """预处理特征数据"""
    processed_data = data.copy()
    
    # 检查特征是否存在
    missing_features = [f for f in features if f not in processed_data.columns]
    if missing_features:
        print(f"⚠️  缺失特征: {missing_features}")
        return None
    
    # 提取选择的特征
    feature_data = processed_data[features].copy()
    
    # 处理缺失值
    print(f"📊 特征缺失值统计:")
    for feature in features:
        missing_count = feature_data[feature].isna().sum()
        print(f"  {feature}: {missing_count} 个缺失值")
        
        # 对于数值型特征，用中位数填充缺失值
        if missing_count > 0:
            median_val = feature_data[feature].median()
            if pd.isna(median_val):  # 如果中位数也是NaN（全部都是缺失值）
                # 使用0填充或者特征的典型值
                if feature == 'ip.proto':
                    fill_val = 6  # TCP协议
                elif feature == 'tcp.srcport':
                    fill_val = 0  # 默认端口
                else:
                    fill_val = 0  # 通用默认值
                print(f"    特征全部缺失，使用默认值 {fill_val} 填充")
            else:
                fill_val = median_val
                print(f"    已用中位数 {fill_val} 填充")
            
            feature_data[feature].fillna(fill_val, inplace=True)
    
    return feature_data

# 1. 加载训练集数据
print("\n1️⃣ 加载训练集数据...")
train_normal = load_data_from_directory(os.path.join(train_path, "APP_0"), 0)  # 正常流量
train_pcdn = load_data_from_directory(os.path.join(train_path, "APP_1"), 1)    # PCDN流量

# 2. 加载验证集数据
print("\n2️⃣ 加载验证集数据...")
val_normal = load_data_from_directory(os.path.join(val_path, "APP_0"), 0)
val_pcdn = load_data_from_directory(os.path.join(val_path, "APP_1"), 1)

# 3. 加载测试集数据
print("\n3️⃣ 加载测试集数据...")
test_normal = load_data_from_directory(os.path.join(test_path, "APP_0"), 0)
test_pcdn = load_data_from_directory(os.path.join(test_path, "APP_1"), 1)

# 4. 合并数据集
print("\n4️⃣ 合并数据集...")

train_data = safe_concat([train_normal, train_pcdn], "训练集")
val_data = safe_concat([val_normal, val_pcdn], "验证集")
test_data = safe_concat([test_normal, test_pcdn], "测试集")

# 检查数据集是否为空
if train_data.empty or val_data.empty or test_data.empty:
    print("❌ 数据集为空，请检查数据路径和文件")
    exit()

print(f"训练集: {len(train_data)} 样本 (正常: {len(train_normal)}, PCDN: {len(train_pcdn)})")
print(f"验证集: {len(val_data)} 样本 (正常: {len(val_normal)}, PCDN: {len(val_pcdn)})")
print(f"测试集: {len(test_data)} 样本 (正常: {len(test_normal)}, PCDN: {len(test_pcdn)})")

# 5. 特征提取和预处理
print(f"\n5️⃣ 特征提取和预处理...")
print(f"选择的特征: {selected_features}")

# 预处理各数据集的特征
X_train = preprocess_features(train_data, selected_features)
X_val = preprocess_features(val_data, selected_features)
X_test = preprocess_features(test_data, selected_features)

if X_train is None or X_val is None or X_test is None:
    print("❌ 特征预处理失败，请检查特征名称")
    exit()

# 提取标签
y_train = train_data['label'].values
y_val = val_data['label'].values  
y_test = test_data['label'].values

# 6. 特征标准化（对于决策树可选，但为了一致性保留）
print(f"\n6️⃣ 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只在训练集上fit
X_val_scaled = scaler.transform(X_val)          # 验证集使用训练集的参数
X_test_scaled = scaler.transform(X_test)        # 测试集使用训练集的参数

print(f"✅ 标准化完成")
print(f"训练集特征形状: {X_train_scaled.shape}")
print(f"验证集特征形状: {X_val_scaled.shape}")
print(f"测试集特征形状: {X_test_scaled.shape}")

# 7. 数据集基本统计
print(f"\n7️⃣ 数据集统计信息...")
print(f"训练集标签分布: {np.bincount(y_train)}")
print(f"验证集标签分布: {np.bincount(y_val)}")
print(f"测试集标签分布: {np.bincount(y_test)}")

# 显示特征统计
print(f"\n📊 特征统计 (训练集):")
feature_stats = pd.DataFrame(X_train, columns=selected_features).describe()
print(feature_stats)

print(f"\n🎯 数据准备完成，开始决策树模型训练...")

# 8. 决策树模型训练（带超参数优化）
print(f"\n8️⃣ 开始决策树模型训练...")
print("=" * 80)

# 创建输出目录
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ 创建输出目录: {output_dir}")

# 定义超参数网格
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

print("🔍 开始超参数网格搜索...")
print(f"🔧 搜索空间: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion'])} 组合")

# 创建决策树分类器
dt_base = DecisionTreeClassifier(random_state=42)

# 使用网格搜索找到最佳参数
grid_search = GridSearchCV(
    dt_base, 
    param_grid, 
    cv=3,  # 3折交叉验证
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# 在训练集上进行网格搜索
grid_search.fit(X_train_scaled, y_train)

# 获取最佳模型
dt_model = grid_search.best_estimator_

print(f"✅ 网格搜索完成!")
print(f"🏆 最佳参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"📊 最佳交叉验证得分: {grid_search.best_score_:.4f}")

print("=" * 80)

# 9. 模型预测
print(f"\n9️⃣ 模型预测...")

# 在各数据集上进行预测
y_train_pred = dt_model.predict(X_train_scaled)
y_train_prob = dt_model.predict_proba(X_train_scaled)[:, 1]

y_val_pred = dt_model.predict(X_val_scaled)
y_val_prob = dt_model.predict_proba(X_val_scaled)[:, 1]

y_test_pred = dt_model.predict(X_test_scaled)
y_test_prob = dt_model.predict_proba(X_test_scaled)[:, 1]

# 10. 模型评估
print(f"\n🔟 模型评估结果...")

# 准确率
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# AUC (安全计算，处理单类别情况)
def safe_auc_score(y_true, y_prob, set_name):
    """安全计算AUC，处理单类别情况"""
    if len(np.unique(y_true)) < 2:
        print(f"⚠️  {set_name} 只有一个类别，无法计算AUC")
        return 0.5  # 返回默认值
    return roc_auc_score(y_true, y_prob)

train_auc = safe_auc_score(y_train, y_train_prob, "训练集")
val_auc = safe_auc_score(y_val, y_val_prob, "验证集")
test_auc = safe_auc_score(y_test, y_test_prob, "测试集")

print(f"📊 准确率 (Accuracy):")
print(f"  训练集: {train_acc:.4f}")
print(f"  验证集: {val_acc:.4f}")
print(f"  测试集: {test_acc:.4f}")

print(f"\n📊 AUC值:")
print(f"  训练集: {train_auc:.4f}")
print(f"  验证集: {val_auc:.4f}")
print(f"  测试集: {test_auc:.4f}")

# 详细分类报告
print(f"\n📋 测试集详细分类报告:")
print(classification_report(y_test, y_test_pred, target_names=['Normal Traffic', 'PCDN Traffic']))

# 11. 特征重要性分析
print(f"\n1️⃣1️⃣ 特征重要性分析...")

# 获取特征重要性
feature_importance = dt_model.feature_importances_
feature_names = selected_features

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"📊 特征重要性排序:")
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 12. 决策树结构分析
print(f"\n1️⃣2️⃣ 决策树结构分析...")

print(f"🌳 决策树信息:")
print(f"  树的深度: {dt_model.get_depth()}")
print(f"  叶子节点数: {dt_model.get_n_leaves()}")
print(f"  总节点数: {dt_model.tree_.node_count}")

# 打印决策树规则（简化版）
print(f"\n📋 决策树规则 (前10条):")
tree_rules = export_text(dt_model, feature_names=selected_features, max_depth=3)
print(tree_rules)

print(f"\n🎯 开始生成可视化图表...")

# 13. 可视化结果
print(f"\n1️⃣3️⃣ 生成可视化图表...")

# 创建图表 (3x2布局，包含决策树图)
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Decision Tree PCDN Traffic Classification Results', fontsize=16, fontweight='bold')

# 1. 决策树可视化 (占用两个位置)
ax1 = plt.subplot(3, 2, (1, 2))
plot_tree(dt_model, 
          feature_names=selected_features,
          class_names=['Normal Traffic', 'PCDN Traffic'],
          filled=True,
          max_depth=3,  # 限制显示深度以保持清晰
          fontsize=10,
          ax=ax1)
ax1.set_title('Decision Tree Visualization (max_depth=3)', fontweight='bold', fontsize=14)

# 2. 特征重要性图
ax2 = plt.subplot(3, 2, 3)
colors = plt.cm.Set3(np.linspace(0, 1, len(importance_df)))
bars = ax2.bar(importance_df['feature'], importance_df['importance'], 
               color=colors)
ax2.set_title('Feature Importance Analysis', fontweight='bold')
ax2.set_xlabel('Feature Names')
ax2.set_ylabel('Importance Score')
ax2.tick_params(axis='x', rotation=45)

# 在柱状图上添加数值标签
for bar, importance in zip(bars, importance_df['importance']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. ROC曲线
ax3 = plt.subplot(3, 2, 4)

# 安全绘制ROC曲线
def safe_plot_roc(y_true, y_prob, label, ax):
    """安全绘制ROC曲线"""
    if len(np.unique(y_true)) < 2:
        return  # 跳过单类别情况
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, label=label, linewidth=2)

safe_plot_roc(y_train, y_train_prob, f'Training Set (AUC = {train_auc:.3f})', ax3)
safe_plot_roc(y_val, y_val_prob, f'Validation Set (AUC = {val_auc:.3f})', ax3)
safe_plot_roc(y_test, y_test_prob, f'Test Set (AUC = {test_auc:.3f})', ax3)

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax3.set_title('ROC Curve Comparison', fontweight='bold')
ax3.set_xlabel('False Positive Rate (FPR)')
ax3.set_ylabel('True Positive Rate (TPR)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 混淆矩阵
ax4 = plt.subplot(3, 2, 5)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal Traffic', 'PCDN Traffic'],
            yticklabels=['Normal Traffic', 'PCDN Traffic'])
ax4.set_title('Test Set Confusion Matrix', fontweight='bold')
ax4.set_xlabel('Predicted Label')
ax4.set_ylabel('True Label')

# 5. 准确率对比
ax5 = plt.subplot(3, 2, 6)
datasets = ['Training Set', 'Validation Set', 'Test Set']
accuracies = [train_acc, val_acc, test_acc]
colors_acc = ['#FF9999', '#66B2FF', '#99FF99']

bars = ax5.bar(datasets, accuracies, color=colors_acc, alpha=0.8)
ax5.set_title('Accuracy Comparison Across Datasets', fontweight='bold')
ax5.set_ylabel('Accuracy')
ax5.set_ylim(0, 1.1)

# 在柱状图上添加数值标签
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# 安全保存图表
try:
    plt.savefig(os.path.join(output_dir, 'decision_tree_classification_results.png'), 
                dpi=300, bbox_inches='tight')
    print("📊 图表已保存为: output/decision_tree_classification_results.png")
except Exception as e:
    print(f"⚠️  图表保存失败: {e}")
    print("📊 图表仍在内存中显示")

plt.show()

# 14. 决策树详细可视化（单独保存）
print(f"\n1️⃣4️⃣ 生成详细决策树图...")

fig_tree, ax_tree = plt.subplots(1, 1, figsize=(25, 15))
plot_tree(dt_model, 
          feature_names=selected_features,
          class_names=['Normal Traffic', 'PCDN Traffic'],
          filled=True,
          rounded=True,
          fontsize=12,
          ax=ax_tree)
ax_tree.set_title(f'Complete Decision Tree (depth={dt_model.get_depth()}, nodes={dt_model.tree_.node_count})', 
                  fontweight='bold', fontsize=16)

try:
    plt.savefig(os.path.join(output_dir, 'decision_tree_detailed.png'), 
                dpi=300, bbox_inches='tight')
    print("📊 详细决策树图已保存为: output/decision_tree_detailed.png")
except Exception as e:
    print(f"⚠️  详细决策树图保存失败: {e}")

plt.show()

# 15. 总结报告
print(f"\n1️⃣5️⃣ 决策树分类任务总结报告")
print("=" * 60)
print(f"🎯 任务: PCDN流量与正常流量二分类")
print(f"🌳 模型: Decision Tree Classifier")
print(f"🔧 使用特征: {', '.join(selected_features)}")
print(f"📊 数据规模:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本") 
print(f"  测试集: {len(X_test)} 样本")

print(f"\n🏆 最终性能指标:")
print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  测试集AUC值: {test_auc:.4f}")

print(f"\n🌳 决策树模型参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n🌳 决策树结构:")
print(f"  树的深度: {dt_model.get_depth()}")
print(f"  叶子节点数: {dt_model.get_n_leaves()}")
print(f"  总节点数: {dt_model.tree_.node_count}")

print(f"\n📈 特征重要性:")
for idx, row in importance_df.iterrows():
    percentage = (row['importance'] / importance_df['importance'].sum()) * 100
    print(f"  {row['feature']}: {row['importance']:.4f} ({percentage:.1f}%)")

print(f"\n📁 输出文件:")
print(f"  主要结果图: output/decision_tree_classification_results.png")
print(f"  详细决策树: output/decision_tree_detailed.png")

print(f"\n✅ 决策树分析完成!")
print("=" * 60)
