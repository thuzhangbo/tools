# PCDN vs Normal Traffic Classification using Decision Tree
# 使用 ip.proto 和 tcp.srcport 特征进行决策树二分类
#
# 🚀 数据缓存功能说明:
# - 首次运行: 加载原始CSV数据，预处理后自动保存到缓存
# - 后续运行: 自动检测并加载缓存数据，大幅提升启动速度
# - 强制重新加载: 将下方 force_reload 设置为 True
# - 缓存位置: data_cache/preprocessed_data.pkl

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
import pickle
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

# 数据缓存设置
cache_dir = "data_cache"
cache_file = os.path.join(cache_dir, "preprocessed_data.pkl")
force_reload = False  # 设置为True强制重新加载数据

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

# 检查是否存在缓存数据
if not force_reload and os.path.exists(cache_file):
    print(f"\n🚀 发现缓存数据，正在快速加载...")
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        X_train_scaled = cached_data['X_train_scaled']
        X_val_scaled = cached_data['X_val_scaled']  
        X_test_scaled = cached_data['X_test_scaled']
        y_train = cached_data['y_train']
        y_val = cached_data['y_val']
        y_test = cached_data['y_test']
        scaler = cached_data['scaler']
        
        print(f"✅ 缓存数据加载成功!")
        print(f"📊 训练集: {X_train_scaled.shape[0]} 样本, {X_train_scaled.shape[1]} 特征")
        print(f"📊 验证集: {X_val_scaled.shape[0]} 样本")
        print(f"📊 测试集: {X_test_scaled.shape[0]} 样本")
        print(f"🎯 跳转到模型训练...")
        
        # 跳转到模型训练部分
        data_loaded_from_cache = True
        
    except Exception as e:
        print(f"⚠️  缓存加载失败: {e}")
        print(f"🔄 将重新加载原始数据...")
        data_loaded_from_cache = False
else:
    data_loaded_from_cache = False

if not data_loaded_from_cache:
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
    
    # 7. 保存预处理后的数据到缓存
    print(f"\n💾 保存预处理后的数据到缓存...")
    try:
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # 准备要保存的数据
        cache_data = {
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'selected_features': selected_features,
            'data_shapes': {
                'train': X_train_scaled.shape,
                'val': X_val_scaled.shape,
                'test': X_test_scaled.shape
            }
        }
        
        # 保存到pickle文件
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"✅ 数据缓存保存成功: {cache_file}")
        print(f"📁 缓存大小: {os.path.getsize(cache_file) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"⚠️  数据缓存保存失败: {e}")
        print(f"🔄 继续正常流程...")

# 无论是从缓存加载还是重新处理，都继续执行数据统计
print(f"\n8️⃣ 数据集统计信息...")
print(f"训练集特征形状: {X_train_scaled.shape}")
print(f"验证集特征形状: {X_val_scaled.shape}")
print(f"测试集特征形状: {X_test_scaled.shape}")
print(f"训练集标签分布: {np.bincount(y_train)}")
print(f"验证集标签分布: {np.bincount(y_val)}")
print(f"测试集标签分布: {np.bincount(y_test)}")

# 显示特征统计（从原始数据重新创建）
if not data_loaded_from_cache:
    print(f"\n📊 特征统计 (训练集):")
    feature_stats = pd.DataFrame(X_train, columns=selected_features).describe()
    print(feature_stats)
else:
    print(f"\n📊 使用特征: {selected_features}")
    print(f"🚀 数据已从缓存加载，跳过详细统计")

print(f"\n🎯 数据准备完成，开始决策树模型训练...")

# 9. 决策树模型训练（带超参数优化）
print(f"\n9️⃣ 开始决策树模型训练...")
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

# 10. 模型预测
print(f"\n🔟 模型预测...")

# 在各数据集上进行预测
y_train_pred = dt_model.predict(X_train_scaled)
y_train_prob = dt_model.predict_proba(X_train_scaled)[:, 1]

y_val_pred = dt_model.predict(X_val_scaled)
y_val_prob = dt_model.predict_proba(X_val_scaled)[:, 1]

y_test_pred = dt_model.predict(X_test_scaled)
y_test_prob = dt_model.predict_proba(X_test_scaled)[:, 1]

# 11. 模型评估
print(f"\n1️⃣1️⃣ 模型评估结果...")

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

# 12. 特征重要性分析
print(f"\n1️⃣2️⃣ 特征重要性分析...")

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

# 13. 决策树结构分析
print(f"\n1️⃣3️⃣ 决策树结构分析...")

print(f"🌳 决策树信息:")
print(f"  树的深度: {dt_model.get_depth()}")
print(f"  叶子节点数: {dt_model.get_n_leaves()}")
print(f"  总节点数: {dt_model.tree_.node_count}")

# 打印决策树规则（简化版）
print(f"\n📋 决策树规则 (前10条):")
tree_rules = export_text(dt_model, feature_names=selected_features, max_depth=3)
print(tree_rules)

print(f"\n🎯 开始生成可视化图表...")

# 14. 可视化结果
print(f"\n1️⃣4️⃣ 生成可视化图表...")

# 首先创建决策树的单独可视化图
print("🌳 生成决策树单独可视化图...")

# 确定合适的显示深度（平衡清晰度和信息量）
tree_depth = dt_model.get_depth()
display_depth = min(tree_depth, 4)  # 最多显示4层以保持清晰

fig_tree_single, ax_tree_single = plt.subplots(1, 1, figsize=(24, 16))

# 使用更好的配色和样式
plot_tree(dt_model, 
          feature_names=selected_features,
          class_names=['Normal Traffic', 'PCDN Traffic'],
          filled=True,
          rounded=True,
          max_depth=display_depth,
          fontsize=14,
          proportion=True,  # 显示比例信息
          impurity=True,    # 显示不纯度
          ax=ax_tree_single)

# 设置标题和样式
title_text = f'Decision Tree for PCDN Traffic Classification\n'
title_text += f'(Showing top {display_depth} levels, Total depth: {tree_depth}, Total nodes: {dt_model.tree_.node_count})'
ax_tree_single.set_title(title_text, fontweight='bold', fontsize=18, pad=20)

# 移除坐标轴
ax_tree_single.set_xticks([])
ax_tree_single.set_yticks([])
ax_tree_single.spines['top'].set_visible(False)
ax_tree_single.spines['right'].set_visible(False)
ax_tree_single.spines['bottom'].set_visible(False)
ax_tree_single.spines['left'].set_visible(False)

# 添加图例说明
legend_text = """
📋 How to Read This Decision Tree:

🔹 Node Information:
   • Feature condition: [feature ≤ threshold]
   • gini: Impurity measure (0.0 = pure, 0.5 = mixed)
   • samples: Number of training samples reaching this node
   • value: [Normal Traffic count, PCDN Traffic count]
   • class: Final prediction for this node

🔹 Colors:
   • Orange tones: Predominantly Normal Traffic
   • Blue tones: Predominantly PCDN Traffic
   • Darker = more confident, Lighter = more mixed

🔹 Decision Path:
   • Follow Yes (True) → Left branch
   • Follow No (False) → Right branch
   • Leaf nodes show final classification
"""

ax_tree_single.text(0.02, 0.02, legend_text, transform=ax_tree_single.transAxes, 
                   fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

plt.tight_layout()

# 保存高清的决策树图
try:
    plt.savefig(os.path.join(output_dir, 'decision_tree_single_clear.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 清晰决策树图已保存为: output/decision_tree_single_clear.png")
except Exception as e:
    print(f"⚠️  决策树图保存失败: {e}")

plt.show()

# 创建其他分析图表 (2x2布局，不包含决策树)
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Decision Tree PCDN Traffic Classification - Performance Analysis', fontsize=16, fontweight='bold')

# 1. 特征重要性图
ax1 = plt.subplot(2, 2, 1)
colors = plt.cm.Set3(np.linspace(0, 1, len(importance_df)))
bars = ax1.bar(importance_df['feature'], importance_df['importance'], 
               color=colors)
ax1.set_title('Feature Importance Analysis', fontweight='bold')
ax1.set_xlabel('Feature Names')
ax1.set_ylabel('Importance Score')
ax1.tick_params(axis='x', rotation=45)

# 在柱状图上添加数值标签
for bar, importance in zip(bars, importance_df['importance']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. ROC曲线
ax2 = plt.subplot(2, 2, 2)

# 安全绘制ROC曲线
def safe_plot_roc(y_true, y_prob, label, ax):
    """安全绘制ROC曲线"""
    if len(np.unique(y_true)) < 2:
        return  # 跳过单类别情况
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, label=label, linewidth=2)

safe_plot_roc(y_train, y_train_prob, f'Training Set (AUC = {train_auc:.3f})', ax2)
safe_plot_roc(y_val, y_val_prob, f'Validation Set (AUC = {val_auc:.3f})', ax2)
safe_plot_roc(y_test, y_test_prob, f'Test Set (AUC = {test_auc:.3f})', ax2)

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax2.set_title('ROC Curve Comparison', fontweight='bold')
ax2.set_xlabel('False Positive Rate (FPR)')
ax2.set_ylabel('True Positive Rate (TPR)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 混淆矩阵
ax3 = plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Normal Traffic', 'PCDN Traffic'],
            yticklabels=['Normal Traffic', 'PCDN Traffic'])
ax3.set_title('Test Set Confusion Matrix', fontweight='bold')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')

# 4. 准确率对比
ax4 = plt.subplot(2, 2, 4)
datasets = ['Training Set', 'Validation Set', 'Test Set']
accuracies = [train_acc, val_acc, test_acc]
colors_acc = ['#FF9999', '#66B2FF', '#99FF99']

bars = ax4.bar(datasets, accuracies, color=colors_acc, alpha=0.8)
ax4.set_title('Accuracy Comparison Across Datasets', fontweight='bold')
ax4.set_ylabel('Accuracy')
ax4.set_ylim(0, 1.1)

# 在柱状图上添加数值标签
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# 安全保存图表
try:
    plt.savefig(os.path.join(output_dir, 'decision_tree_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print("📊 性能分析图已保存为: output/decision_tree_performance_analysis.png")
except Exception as e:
    print(f"⚠️  性能分析图保存失败: {e}")
    print("📊 图表仍在内存中显示")

plt.show()

# 14.5. 阈值优化可视化
print(f"\n1️⃣4️⃣.5️⃣ 生成阈值优化可视化图表...")

# 首先定义阈值分析函数
def threshold_analysis(y_true, y_prob, set_name="Test Set"):
    """
    阈值分析函数，寻找最优分类阈值
    """
    from sklearn.metrics import precision_recall_curve
    
    print(f"\n🔍 {set_name} 阈值分析:")
    
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # 计算F1分数
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    # 寻找最佳阈值的不同策略
    strategies = {}
    
    # 策略1: 最大F1分数
    max_f1_idx = np.argmax(f1_scores)
    strategies['Max F1'] = {
        'threshold': thresholds[max_f1_idx],
        'f1': f1_scores[max_f1_idx],
        'precision': precision[max_f1_idx],
        'recall': recall[max_f1_idx]
    }
    
    # 策略2: 高精确率 (降低误报率，精确率 >= 0.95)
    high_precision_mask = precision[:-1] >= 0.95
    if np.any(high_precision_mask):
        high_prec_idx = np.where(high_precision_mask)[0][-1]  # 取最后一个满足条件的
        strategies['High Precision (≥95%)'] = {
            'threshold': thresholds[high_prec_idx],
            'f1': f1_scores[high_prec_idx],
            'precision': precision[high_prec_idx],
            'recall': recall[high_prec_idx]
        }
    
    # 策略3: 平衡精确率和召回率 (precision >= 0.90)
    balanced_mask = precision[:-1] >= 0.90
    if np.any(balanced_mask):
        # 在满足精确率要求的前提下，选择F1最大的
        balanced_indices = np.where(balanced_mask)[0]
        best_balanced_idx = balanced_indices[np.argmax(f1_scores[balanced_indices])]
        strategies['Balanced (Precision ≥90%)'] = {
            'threshold': thresholds[best_balanced_idx],
            'f1': f1_scores[best_balanced_idx],
            'precision': precision[best_balanced_idx],
            'recall': recall[best_balanced_idx]
        }
    
    # 策略4: 固定误报率 (FPR <= 0.05, 即95%的正常流量正确分类)
    # 计算不同阈值下的误报率
    fpr_list = []
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fpr = fp / (fp + tn + 1e-8)
        fpr_list.append(fpr)
    
    fpr_array = np.array(fpr_list)
    low_fpr_mask = fpr_array <= 0.05
    if np.any(low_fpr_mask):
        low_fpr_idx = np.where(low_fpr_mask)[0][0]  # 取第一个满足条件的
        strategies['Low False Positive (FPR ≤5%)'] = {
            'threshold': thresholds[low_fpr_idx],
            'f1': f1_scores[low_fpr_idx],
            'precision': precision[low_fpr_idx],
            'recall': recall[low_fpr_idx],
            'fpr': fpr_array[low_fpr_idx]
        }
    
    return strategies, precision, recall, thresholds, f1_scores

# 创建阈值分析图表
fig_threshold = plt.figure(figsize=(20, 12))
fig_threshold.suptitle('Threshold Optimization Analysis - Reducing False Positive Rate', fontsize=16, fontweight='bold')

# 计算精确率-召回率曲线数据
test_precision_full, test_recall_full, test_thresholds_full = precision_recall_curve(y_test, y_test_prob)
test_f1_scores_full = 2 * (test_precision_full[:-1] * test_recall_full[:-1]) / (test_precision_full[:-1] + test_recall_full[:-1] + 1e-8)

# 计算误报率曲线
test_fpr_list = []
for threshold in test_thresholds_full:
    y_pred_thresh = (y_test_prob >= threshold).astype(int)
    tn = np.sum((y_test == 0) & (y_pred_thresh == 0))
    fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
    fpr = fp / (fp + tn + 1e-8)
    test_fpr_list.append(fpr)

test_fpr_array = np.array(test_fpr_list)

# 1. 精确率-召回率曲线
ax1 = plt.subplot(2, 3, 1)
ax1.plot(test_recall_full, test_precision_full, 'b-', linewidth=2, label='PR Curve')
ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Precision')
ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Precision')
ax1.set_xlabel('Recall (Sensitivity)')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. F1分数vs阈值
ax2 = plt.subplot(2, 3, 2)
ax2.plot(test_thresholds_full, test_f1_scores_full, 'g-', linewidth=2, label='F1 Score')
max_f1_idx = np.argmax(test_f1_scores_full)
ax2.scatter(test_thresholds_full[max_f1_idx], test_f1_scores_full[max_f1_idx], 
           color='red', s=100, zorder=5, label=f'Max F1 (t={test_thresholds_full[max_f1_idx]:.3f})')
ax2.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Default (0.5)')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score vs Threshold', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. 误报率vs阈值
ax3 = plt.subplot(2, 3, 3)
ax3.plot(test_thresholds_full, test_fpr_array, 'r-', linewidth=2, label='False Positive Rate')
ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% FPR Target')
ax3.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Default (0.5)')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('False Positive Rate')
ax3.set_title('False Positive Rate vs Threshold', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. 精确率vs阈值
ax4 = plt.subplot(2, 3, 4)
ax4.plot(test_thresholds_full, test_precision_full[:-1], 'purple', linewidth=2, label='Precision')
ax4.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Target')
ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
ax4.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Default (0.5)')
ax4.set_xlabel('Threshold')
ax4.set_ylabel('Precision')
ax4.set_title('Precision vs Threshold', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. 召回率vs阈值
ax5 = plt.subplot(2, 3, 5)
ax5.plot(test_thresholds_full, test_recall_full[:-1], 'brown', linewidth=2, label='Recall')
ax5.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Default (0.5)')
ax5.set_xlabel('Threshold')
ax5.set_ylabel('Recall (Sensitivity)')
ax5.set_title('Recall vs Threshold', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. 阈值策略对比
ax6 = plt.subplot(2, 3, 6)

# 预先计算策略数据，避免变量未定义问题
temp_strategies, _, _, _, _ = threshold_analysis(y_test, y_test_prob, "测试集（预计算）")

if temp_strategies:
    strategy_names = list(temp_strategies.keys())
    precision_values = [temp_strategies[name]['precision'] for name in strategy_names]
    recall_values = [temp_strategies[name]['recall'] for name in strategy_names]
    f1_values = [temp_strategies[name]['f1'] for name in strategy_names]
    
    x_pos = np.arange(len(strategy_names))
    width = 0.25
    
    bars1 = ax6.bar(x_pos - width, precision_values, width, label='Precision', alpha=0.8)
    bars2 = ax6.bar(x_pos, recall_values, width, label='Recall', alpha=0.8) 
    bars3 = ax6.bar(x_pos + width, f1_values, width, label='F1 Score', alpha=0.8)
    
    ax6.set_xlabel('Strategy')
    ax6.set_ylabel('Score')
    ax6.set_title('Strategy Comparison', fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([name.replace(' ', '\n') for name in strategy_names], fontsize=9)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

plt.tight_layout()

# 保存阈值分析图
try:
    plt.savefig(os.path.join(output_dir, 'threshold_optimization_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print("📊 阈值优化分析图已保存为: output/threshold_optimization_analysis.png")
except Exception as e:
    print(f"⚠️  阈值分析图保存失败: {e}")

plt.show()

# 15. 阈值优化分析 - 降低误报率
print(f"\n1️⃣5️⃣ 阈值优化分析 - 降低正常流量误分类...")
print("=" * 80)

from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


# 对测试集进行阈值分析
print("🎯 重点分析: 如何降低正常流量被误分类为PCDN流量的情况")

test_strategies, test_precision, test_recall, test_thresholds, test_f1_scores = threshold_analysis(y_test, y_test_prob, "测试集")

print(f"\n📊 不同阈值策略的性能对比:")
print(f"{'策略':<25} {'阈值':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'说明'}")
print("-" * 80)

strategy_descriptions = {
    'Max F1': '最大F1分数（平衡精确率和召回率）',
    'High Precision (≥95%)': '高精确率策略（最小化误报）',
    'Balanced (Precision ≥90%)': '平衡策略（90%以上精确率）',
    'Low False Positive (FPR ≤5%)': '低误报率策略（≤5%正常流量误分类）'
}

for strategy_name, metrics in test_strategies.items():
    desc = strategy_descriptions.get(strategy_name, '')
    print(f"{strategy_name:<25} {metrics['threshold']:<8.3f} {metrics['precision']:<8.3f} "
          f"{metrics['recall']:<8.3f} {metrics['f1']:<8.3f} {desc}")

# 详细分析默认阈值vs最优阈值
print(f"\n🔍 默认阈值(0.5) vs 低误报阈值 详细对比:")

# 默认阈值性能
default_threshold = 0.5
y_test_pred_default = (y_test_prob >= default_threshold).astype(int)
default_precision = precision_score(y_test, y_test_pred_default)
default_recall = recall_score(y_test, y_test_pred_default)
default_f1 = f1_score(y_test, y_test_pred_default)

# 计算默认阈值的混淆矩阵
from sklearn.metrics import confusion_matrix as cm
default_cm = cm(y_test, y_test_pred_default)
default_tn, default_fp, default_fn, default_tp = default_cm.ravel()
default_fpr = default_fp / (default_fp + default_tn)

print(f"\n📈 默认阈值 (0.5) 性能:")
print(f"  精确率: {default_precision:.4f}")
print(f"  召回率: {default_recall:.4f}")
print(f"  F1分数: {default_f1:.4f}")
print(f"  误报率: {default_fpr:.4f} ({default_fpr*100:.2f}%)")
print(f"  误分类的正常流量: {default_fp} / {default_fp + default_tn} ({default_fpr*100:.1f}%)")

# 如果有低误报率策略，显示对比
if 'Low False Positive (FPR ≤5%)' in test_strategies:
    low_fpr_strategy = test_strategies['Low False Positive (FPR ≤5%)']
    optimal_threshold = low_fpr_strategy['threshold']
    
    y_test_pred_optimal = (y_test_prob >= optimal_threshold).astype(int)
    optimal_cm = cm(y_test, y_test_pred_optimal)
    optimal_tn, optimal_fp, optimal_fn, optimal_tp = optimal_cm.ravel()
    optimal_fpr = optimal_fp / (optimal_fp + optimal_tn)
    
    print(f"\n🎯 低误报阈值 ({optimal_threshold:.3f}) 性能:")
    print(f"  精确率: {low_fpr_strategy['precision']:.4f}")
    print(f"  召回率: {low_fpr_strategy['recall']:.4f}")
    print(f"  F1分数: {low_fpr_strategy['f1']:.4f}")
    print(f"  误报率: {optimal_fpr:.4f} ({optimal_fpr*100:.2f}%)")
    print(f"  误分类的正常流量: {optimal_fp} / {optimal_fp + optimal_tn} ({optimal_fpr*100:.1f}%)")
    
    print(f"\n✅ 改进效果:")
    fp_reduction = default_fp - optimal_fp
    fpr_reduction = default_fpr - optimal_fpr
    print(f"  误分类正常流量减少: {fp_reduction} 个样本")
    print(f"  误报率降低: {fpr_reduction:.4f} ({fpr_reduction*100:.2f} 个百分点)")
    if default_fp > 0:
        print(f"  误报率相对降低: {(fp_reduction/default_fp)*100:.1f}%")

# 16. 生成完整的决策树文本规则
print(f"\n1️⃣6️⃣ 生成完整决策树规则...")

# 导出完整的决策树规则到文件
try:
    full_tree_rules = export_text(dt_model, feature_names=selected_features)
    rules_file = os.path.join(output_dir, 'decision_tree_rules.txt')
    with open(rules_file, 'w', encoding='utf-8') as f:
        f.write("Decision Tree Rules for PCDN Traffic Classification\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Parameters:\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nTree Structure:\n")
        f.write(f"  Tree Depth: {dt_model.get_depth()}\n")
        f.write(f"  Number of Leaves: {dt_model.get_n_leaves()}\n")
        f.write(f"  Total Nodes: {dt_model.tree_.node_count}\n\n")
        
        # 添加阈值优化信息
        if test_strategies:
            f.write("Threshold Optimization Results:\n")
            f.write("-" * 40 + "\n")
            f.write("Recommended thresholds to reduce false positive rate:\n\n")
            for strategy_name, metrics in test_strategies.items():
                f.write(f"{strategy_name}:\n")
                f.write(f"  Threshold: {metrics['threshold']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                if 'fpr' in metrics:
                    f.write(f"  False Positive Rate: {metrics['fpr']:.4f}\n")
                f.write("\n")
            
            f.write("Usage Recommendations:\n")
            f.write("- For minimum false positives: Use High Precision strategy\n")
            f.write("- For balanced performance: Use Max F1 strategy\n")
            f.write("- For production deployment: Choose threshold based on business requirements\n\n")
        
        f.write("Decision Rules:\n")
        f.write("-" * 40 + "\n")
        f.write(full_tree_rules)
    
    print(f"📝 完整决策树规则已保存为: output/decision_tree_rules.txt")
    
    # 同时保存阈值优化结果到单独文件
    if test_strategies:
        threshold_report_file = os.path.join(output_dir, 'threshold_optimization_report.txt')
        with open(threshold_report_file, 'w', encoding='utf-8') as f:
            f.write("Threshold Optimization Report for PCDN Traffic Classification\n")
            f.write("=" * 70 + "\n\n")
            f.write("目标: 降低正常流量误分类为PCDN流量的误报率\n\n")
            
            f.write("分析方法:\n")
            f.write("通过调整分类阈值来优化精确率、召回率和F1分数，特别关注降低误报率\n\n")
            
            f.write("推荐阈值策略:\n")
            f.write("-" * 30 + "\n")
            
            for strategy_name, metrics in test_strategies.items():
                f.write(f"\n{strategy_name}:\n")
                f.write(f"  推荐阈值: {metrics['threshold']:.4f}\n")
                f.write(f"  精确率: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)\n")
                f.write(f"  召回率: {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                if 'fpr' in metrics:
                    f.write(f"  误报率: {metrics['fpr']:.4f} ({metrics['fpr']*100:.1f}%)\n")
                
                # 添加应用场景说明
                if 'High Precision' in strategy_name:
                    f.write(f"  适用场景: 对误报极其敏感的生产环境\n")
                elif 'Low False Positive' in strategy_name:
                    f.write(f"  适用场景: 需要控制误报率在5%以下的业务场景\n")
                elif 'Max F1' in strategy_name:
                    f.write(f"  适用场景: 需要平衡精确率和召回率的一般场景\n")
                elif 'Balanced' in strategy_name:
                    f.write(f"  适用场景: 对精确率有一定要求但不过分严格的场景\n")
            
            f.write(f"\n实施建议:\n")
            f.write("-" * 20 + "\n")
            f.write("1. 根据业务对误报的容忍度选择合适的策略\n")
            f.write("2. 在生产环境中先用小流量测试选定阈值的效果\n") 
            f.write("3. 定期重新评估和调整阈值，适应数据分布的变化\n")
            f.write("4. 建议同时监控精确率、召回率和F1分数的变化\n")
        
        print(f"📝 阈值优化报告已保存为: output/threshold_optimization_report.txt")
except Exception as e:
    print(f"⚠️  决策树规则保存失败: {e}")

# 16. 总结报告
print(f"\n1️⃣6️⃣ 决策树分类任务总结报告")
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

print(f"\n🎯 阈值优化结果:")
if test_strategies:
    print(f"  推荐阈值策略:")
    for strategy_name, metrics in test_strategies.items():
        if 'Low False Positive' in strategy_name or 'High Precision' in strategy_name:
            print(f"    {strategy_name}: 阈值={metrics['threshold']:.3f}")
            print(f"      精确率={metrics['precision']:.3f}, 召回率={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

print(f"\n📁 输出文件:")
print(f"  决策树可视化: output/decision_tree_single_clear.png")
print(f"  性能分析图: output/decision_tree_performance_analysis.png")
print(f"  阈值优化分析图: output/threshold_optimization_analysis.png")
print(f"  决策规则文件: output/decision_tree_rules.txt")
print(f"  阈值优化报告: output/threshold_optimization_report.txt")

print(f"\n💡 使用建议:")
print(f"  • 如需最小化误报率: 使用高精确率阈值策略")
print(f"  • 如需平衡性能: 使用最大F1分数阈值")
print(f"  • 实际部署时: 可根据业务需求选择合适的阈值")

print(f"\n✅ 决策树分析与阈值优化完成!")
print("=" * 60)
