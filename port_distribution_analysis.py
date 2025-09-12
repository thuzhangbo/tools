# Field Distribution Analysis for PCDN vs Normal Traffic
# 字段分布分析：正常流量 vs PCDN流量
# 
# 分析内容:
# 1. 端口字段分析 (tcp.srcport, tcp.dstport, udp.srcport, udp.dstport)
#    端口映射规则:
#    - 0-1023: 系统/知名端口 → 0
#    - 1024-49151: 注册端口 → 1  
#    - 49152-65535: 动态/私有端口 → 2
#
# 2. 比例字段分析 (down_byte_ratio)
#    比例区间映射规则 (0.1区间):
#    - [0.0, 0.1) → 0, [0.1, 0.2) → 1, ..., [0.9, 1.0] → 9
#
# 输出: 10张独立图表 (4端口×2流量 + 1比例×2流量)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

print("📊 开始字段分布分析")
print("🎯 分析目标: 正常流量 vs PCDN流量的端口使用模式和比例字段分布")
print("=" * 60)

# 定义数据路径
base_path = "pcdn_32_pkts_2class_feature_enhance_v17.4_dataset"
train_path = os.path.join(base_path, "Training_set")
val_path = os.path.join(base_path, "Validation_set") 
test_path = os.path.join(base_path, "Testing_set")

# 分析的端口字段
port_fields = ['tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']

# 分析的比例字段
ratio_fields = ['down_byte_ratio']

# 端口分类映射函数
def map_port_category(port_value):
    """
    将端口号映射到类别
    0-1023: 系统/知名端口 → 0
    1024-49151: 注册端口 → 1
    49152-65535: 动态/私有端口 → 2
    """
    if pd.isna(port_value):
        return -1  # 缺失值
    
    port_value = int(port_value)
    
    if 0 <= port_value <= 1023:
        return 0  # 系统/知名端口
    elif 1024 <= port_value <= 49151:
        return 1  # 注册端口
    elif 49152 <= port_value <= 65535:
        return 2  # 动态/私有端口
    else:
        return -1  # 无效端口

# 端口类别标签
port_category_labels = {
    0: 'System/Well-known\n(0-1023)',
    1: 'Registered\n(1024-49151)', 
    2: 'Dynamic/Private\n(49152-65535)',
    -1: 'Missing/Invalid'
}

# 比例区间映射函数
def map_ratio_interval(ratio_value):
    """
    将比例值映射到0.1的区间
    [0.0, 0.1) → 0
    [0.1, 0.2) → 1
    ...
    [0.9, 1.0] → 9
    """
    if pd.isna(ratio_value):
        return -1  # 缺失值
    
    try:
        ratio_value = float(ratio_value)
        
        if ratio_value < 0 or ratio_value > 1:
            return -1  # 无效值
        
        # 特殊处理1.0，归到最后一个区间
        if ratio_value == 1.0:
            return 9
        
        # 计算区间索引
        interval_idx = int(ratio_value / 0.1)
        return min(interval_idx, 9)  # 确保不超过9
        
    except (ValueError, TypeError):
        return -1  # 无效值

# 比例区间标签
ratio_interval_labels = {
    0: '[0.0, 0.1)',
    1: '[0.1, 0.2)',
    2: '[0.2, 0.3)',
    3: '[0.3, 0.4)',
    4: '[0.4, 0.5)',
    5: '[0.5, 0.6)',
    6: '[0.6, 0.7)',
    7: '[0.7, 0.8)',
    8: '[0.8, 0.9)',
    9: '[0.9, 1.0]',
    -1: 'Missing/Invalid'
}

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

def safe_concat(dataframes, set_name):
    """安全合并数据集，处理空数据集的情况"""
    non_empty_dfs = [df for df in dataframes if not df.empty]
    if not non_empty_dfs:
        print(f"⚠️  {set_name} 没有有效数据!")
        return pd.DataFrame()
    return pd.concat(non_empty_dfs, ignore_index=True)

# 1. 加载所有数据
print("\n1️⃣ 加载训练集数据...")
train_normal = load_data_from_directory(os.path.join(train_path, "APP_0"), 0)  # 正常流量
train_pcdn = load_data_from_directory(os.path.join(train_path, "APP_1"), 1)    # PCDN流量

print("\n2️⃣ 加载验证集数据...")
val_normal = load_data_from_directory(os.path.join(val_path, "APP_0"), 0)
val_pcdn = load_data_from_directory(os.path.join(val_path, "APP_1"), 1)

print("\n3️⃣ 加载测试集数据...")
test_normal = load_data_from_directory(os.path.join(test_path, "APP_0"), 0)
test_pcdn = load_data_from_directory(os.path.join(test_path, "APP_1"), 1)

# 2. 分别汇总正常流量和PCDN流量
print("\n4️⃣ 汇总流量数据...")

# 汇总正常流量（所有数据集）
normal_traffic_all = safe_concat([train_normal, val_normal, test_normal], "正常流量汇总")

# 汇总PCDN流量（所有数据集）  
pcdn_traffic_all = safe_concat([train_pcdn, val_pcdn, test_pcdn], "PCDN流量汇总")

print(f"📊 正常流量总样本数: {len(normal_traffic_all)}")
print(f"📊 PCDN流量总样本数: {len(pcdn_traffic_all)}")

# 检查端口字段是否存在
missing_port_fields = []
for field in port_fields:
    if field not in normal_traffic_all.columns:
        missing_port_fields.append(field)

if missing_port_fields:
    print(f"⚠️  缺失端口字段: {missing_port_fields}")
    print("🔍 可用的端口相关字段:")
    available_port_fields = [col for col in normal_traffic_all.columns if 'port' in col.lower()]
    for field in available_port_fields[:10]:  # 显示前10个
        print(f"  - {field}")
    
    # 使用实际存在的字段
    port_fields = [field for field in port_fields if field in normal_traffic_all.columns]

# 检查比例字段是否存在
missing_ratio_fields = []
for field in ratio_fields:
    if field not in normal_traffic_all.columns:
        missing_ratio_fields.append(field)

if missing_ratio_fields:
    print(f"⚠️  缺失比例字段: {missing_ratio_fields}")
    print("🔍 可用的比例相关字段:")
    available_ratio_fields = [col for col in normal_traffic_all.columns if 'ratio' in col.lower()]
    for field in available_ratio_fields[:10]:  # 显示前10个
        print(f"  - {field}")
    
    # 使用实际存在的字段
    ratio_fields = [field for field in ratio_fields if field in normal_traffic_all.columns]

if not port_fields and not ratio_fields:
    print("❌ 没有找到有效的分析字段")
    exit()

print(f"📋 将分析的端口字段: {port_fields}")
print(f"📋 将分析的比例字段: {ratio_fields}")

# 3. 字段分类映射
print("\n5️⃣ 进行字段分类映射...")

def analyze_field_distribution(data, traffic_type):
    """分析字段分布（包括端口和比例字段）"""
    results = {}
    
    # 分析端口字段
    for field in port_fields:
        print(f"  分析端口字段 {field} 在 {traffic_type} 中的分布...")
        
        # 应用端口映射
        categories = data[field].apply(map_port_category)
        
        # 统计各类别的数量
        category_counts = categories.value_counts().sort_index()
        
        # 计算百分比
        total_valid = len(categories[categories != -1])
        if total_valid > 0:
            category_percentages = (category_counts / total_valid * 100).round(2)
        else:
            category_percentages = pd.Series()
        
        results[field] = {
            'type': 'port',
            'counts': category_counts,
            'percentages': category_percentages,
            'total_samples': len(data),
            'valid_samples': total_valid,
            'missing_samples': len(categories[categories == -1])
        }
        
        print(f"    总样本: {len(data)}, 有效样本: {total_valid}, 缺失样本: {len(categories[categories == -1])}")
    
    # 分析比例字段
    for field in ratio_fields:
        print(f"  分析比例字段 {field} 在 {traffic_type} 中的分布...")
        
        # 应用比例区间映射
        intervals = data[field].apply(map_ratio_interval)
        
        # 统计各区间的数量
        interval_counts = intervals.value_counts().sort_index()
        
        # 计算百分比
        total_valid = len(intervals[intervals != -1])
        if total_valid > 0:
            interval_percentages = (interval_counts / total_valid * 100).round(2)
        else:
            interval_percentages = pd.Series()
        
        results[field] = {
            'type': 'ratio',
            'counts': interval_counts,
            'percentages': interval_percentages,
            'total_samples': len(data),
            'valid_samples': total_valid,
            'missing_samples': len(intervals[intervals == -1])
        }
        
        print(f"    总样本: {len(data)}, 有效样本: {total_valid}, 缺失样本: {len(intervals[intervals == -1])}")
    
    return results

# 分析正常流量和PCDN流量的字段分布
normal_results = analyze_field_distribution(normal_traffic_all, "正常流量")
pcdn_results = analyze_field_distribution(pcdn_traffic_all, "PCDN流量")

# 4. 生成可视化图表 - 独立图表
print("\n6️⃣ 生成字段分布可视化图表...")

# 创建输出目录
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

traffic_types = ['Normal Traffic', 'PCDN Traffic']
results_data = [normal_results, pcdn_results]
colors = ['skyblue', 'lightcoral']
traffic_names = ['normal', 'pcdn']

saved_charts = []

# 合并所有要分析的字段
all_fields = port_fields + ratio_fields

# 为每个流量类型和字段组合创建独立图表
for traffic_idx, (traffic_type, results, color, traffic_name) in enumerate(zip(traffic_types, results_data, colors, traffic_names)):
    for field in all_fields:
        print(f"  生成图表: {traffic_type} - {field}")
        
        # 创建单独的图表
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if field in results:
            data = results[field]
            field_type = data['type']
            counts = data['counts']
            percentages = data['percentages']
            
            # 根据字段类型选择标签
            if field_type == 'port':
                labels_dict = port_category_labels
                xlabel = 'Port Category'
                possible_values = [0, 1, 2, -1]
            else:  # ratio
                labels_dict = ratio_interval_labels
                xlabel = 'Ratio Interval'
                possible_values = list(range(10)) + [-1]  # 0-9 + -1
            
            # 准备绘图数据
            categories = []
            values = []
            category_labels = []
            
            for cat in possible_values:
                if cat in counts.index:
                    categories.append(cat)
                    values.append(counts[cat])
                    category_labels.append(labels_dict[cat])
                elif cat != -1:  # 对于缺失的有效类别，显示0
                    categories.append(cat)
                    values.append(0)
                    category_labels.append(labels_dict[cat])
            
            # 绘制柱状图
            bars = ax.bar(range(len(categories)), values, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # 设置标题和标签
            ax.set_title(f'{traffic_type} - {field} Distribution', fontweight='bold', fontsize=16, pad=20)
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
            
            # 设置x轴标签
            ax.set_xticks(range(len(categories)))
            if field_type == 'ratio':
                # 比例字段的标签可能较长，旋转45度
                ax.set_xticklabels(category_labels, fontsize=10, ha='right', rotation=45)
            else:
                ax.set_xticklabels(category_labels, fontsize=12, ha='center')
            
            # 在柱子上显示数值和百分比
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    # 显示数量
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.02,
                           f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                    
                    # 显示百分比（如果有有效数据）
                    cat = categories[i]
                    if cat in percentages.index and cat != -1:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.06,
                               f'({percentages[cat]:.1f}%)', ha='center', va='bottom', 
                               fontsize=10, style='italic')
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # 设置y轴范围
            if values:
                ax.set_ylim(0, max(values) * 1.15)
            
            # 添加统计信息文本框
            stats_text = f"Total samples: {data['total_samples']:,}\n"
            stats_text += f"Valid samples: {data['valid_samples']:,}\n"
            stats_text += f"Missing samples: {data['missing_samples']:,}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
        else:
            ax.text(0.5, 0.5, f'No data available for\n{field}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.set_title(f'{traffic_type} - {field} Distribution', fontweight='bold', fontsize=16)
            ax.set_xlabel('Port Category', fontsize=14)
            ax.set_ylabel('Number of Samples', fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表（根据字段类型生成不同的文件名）
        if field in results and results[field]['type'] == 'ratio':
            chart_filename = f'ratio_distribution_{traffic_name}_{field.replace(".", "_")}.png'
        else:
            chart_filename = f'port_distribution_{traffic_name}_{field.replace(".", "_")}.png'
        chart_path = os.path.join(output_dir, chart_filename)
        
        try:
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            saved_charts.append(chart_filename)
            print(f"    ✅ 保存成功: {chart_filename}")
        except Exception as e:
            print(f"    ❌ 保存失败: {chart_filename}, 错误: {e}")
        
        # 显示图表
        plt.show()
        
        # 关闭当前图表以释放内存
        plt.close()

print(f"\n📊 已生成 {len(saved_charts)} 张独立图表:")
for chart in saved_charts:
    print(f"  - output/{chart}")

# 5. 生成详细统计报告
print("\n7️⃣ 生成详细统计报告...")

def generate_detailed_report():
    """生成详细的字段分布报告"""
    report_file = os.path.join(output_dir, 'field_distribution_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Field Distribution Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Analysis Overview:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Port fields analyzed: {len(port_fields)}\n")
        f.write(f"Ratio fields analyzed: {len(ratio_fields)}\n")
        f.write(f"Total charts generated: {len(saved_charts)}\n\n")
        
        f.write("Port Category Mapping:\n")
        f.write("- System/Well-known ports (0-1023) → Category 0\n")
        f.write("- Registered ports (1024-49151) → Category 1\n") 
        f.write("- Dynamic/Private ports (49152-65535) → Category 2\n\n")
        
        f.write("Ratio Interval Mapping (0.1 intervals):\n")
        for i in range(10):
            f.write(f"- {ratio_interval_labels[i]} → Interval {i}\n")
        f.write("\n")
        
        for traffic_type, results in [("Normal Traffic", normal_results), ("PCDN Traffic", pcdn_results)]:
            f.write(f"{traffic_type} Field Distribution:\n")
            f.write("-" * 40 + "\n")
            
            # 端口字段分析
            if port_fields:
                f.write("\nPort Fields:\n")
                for field in port_fields:
                    if field in results:
                        data = results[field]
                        f.write(f"\n{field}:\n")
                        f.write(f"  Total samples: {data['total_samples']:,}\n")
                        f.write(f"  Valid samples: {data['valid_samples']:,}\n")
                        f.write(f"  Missing samples: {data['missing_samples']:,}\n")
                        
                        f.write("  Distribution:\n")
                        for cat, count in data['counts'].items():
                            if cat in data['percentages'].index:
                                percentage = data['percentages'][cat]
                                f.write(f"    {port_category_labels[cat]}: {count:,} ({percentage:.1f}%)\n")
                            else:
                                f.write(f"    {port_category_labels[cat]}: {count:,}\n")
            
            # 比例字段分析
            if ratio_fields:
                f.write("\nRatio Fields:\n")
                for field in ratio_fields:
                    if field in results:
                        data = results[field]
                        f.write(f"\n{field}:\n")
                        f.write(f"  Total samples: {data['total_samples']:,}\n")
                        f.write(f"  Valid samples: {data['valid_samples']:,}\n")
                        f.write(f"  Missing samples: {data['missing_samples']:,}\n")
                        
                        f.write("  Distribution:\n")
                        for cat, count in data['counts'].items():
                            if cat in data['percentages'].index:
                                percentage = data['percentages'][cat]
                                f.write(f"    {ratio_interval_labels[cat]}: {count:,} ({percentage:.1f}%)\n")
                            else:
                                f.write(f"    {ratio_interval_labels[cat]}: {count:,}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
        
        # 对比分析
        f.write("Comparative Analysis:\n")
        f.write("-" * 25 + "\n\n")
        
        # 端口字段对比
        if port_fields:
            f.write("Port Fields Comparison:\n")
            for field in port_fields:
                if field in normal_results and field in pcdn_results:
                    f.write(f"\n{field}:\n")
                    
                    for cat in [0, 1, 2]:
                        normal_pct = normal_results[field]['percentages'].get(cat, 0)
                        pcdn_pct = pcdn_results[field]['percentages'].get(cat, 0)
                        diff = pcdn_pct - normal_pct
                        
                        f.write(f"  {port_category_labels[cat]}:\n")
                        f.write(f"    Normal: {normal_pct:.1f}%, PCDN: {pcdn_pct:.1f}%, Diff: {diff:+.1f}%\n")
        
        # 比例字段对比
        if ratio_fields:
            f.write("\nRatio Fields Comparison:\n")
            for field in ratio_fields:
                if field in normal_results and field in pcdn_results:
                    f.write(f"\n{field}:\n")
                    
                    for cat in range(10):
                        normal_pct = normal_results[field]['percentages'].get(cat, 0)
                        pcdn_pct = pcdn_results[field]['percentages'].get(cat, 0)
                        diff = pcdn_pct - normal_pct
                        
                        if normal_pct > 0 or pcdn_pct > 0:  # 只显示有数据的区间
                            f.write(f"  {ratio_interval_labels[cat]}:\n")
                            f.write(f"    Normal: {normal_pct:.1f}%, PCDN: {pcdn_pct:.1f}%, Diff: {diff:+.1f}%\n")
    
    print(f"📝 详细报告已保存为: output/field_distribution_report.txt")

generate_detailed_report()

# 6. 总结
print("\n8️⃣ 分析总结")
print("=" * 60)
print(f"📊 数据统计:")
print(f"  正常流量样本: {len(normal_traffic_all):,}")
print(f"  PCDN流量样本: {len(pcdn_traffic_all):,}")
print(f"  端口字段数: {len(port_fields)}")
print(f"  比例字段数: {len(ratio_fields)}")
print(f"  总分析字段: {len(all_fields)}")

print(f"\n📁 输出文件:")
print(f"  详细报告: output/field_distribution_report.txt")
print(f"  分布图表 ({len(saved_charts)}张):")
for chart in saved_charts:
    print(f"    - output/{chart}")

print(f"\n🎯 生成的图表类型:")
port_charts = [c for c in saved_charts if c.startswith('port_distribution')]
ratio_charts = [c for c in saved_charts if c.startswith('ratio_distribution')]
print(f"  端口分布图: {len(port_charts)}张")
print(f"  比例分布图: {len(ratio_charts)}张")

print(f"\n✅ 字段分布分析完成!")
print(f"🎉 总共生成了 {len(saved_charts)} 张独立的分布图表!")
print("=" * 60)
