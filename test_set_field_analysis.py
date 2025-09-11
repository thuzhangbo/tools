#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing_set字段分布分析工具
用于检测数据泄露和字段分布异常
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_testing_data():
    """
    加载Testing_set数据
    """
    base_path = "pcdn_32_pkts_2class_feature_enhance_v17.4_dataset/Testing_set"
    
    dataframes = []
    file_info = []
    
    for app_dir in ['APP_0', 'APP_1']:
        app_path = os.path.join(base_path, app_dir)
        if os.path.exists(app_path):
            csv_files = glob.glob(os.path.join(app_path, '*.csv'))
            
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = os.path.basename(file_path)
                    df['app_category'] = app_dir
                    df['label'] = 0 if app_dir == 'APP_0' else 1
                    
                    dataframes.append(df)
                    file_info.append({
                        'file': os.path.basename(file_path),
                        'category': app_dir,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                    print(f"✅ 加载: {os.path.basename(file_path)} ({app_dir}) - {len(df)} 行")
                    
                except Exception as e:
                    print(f"❌ 加载失败: {file_path} - {e}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n📊 Testing_set总览:")
        print(f"  总文件数: {len(file_info)}")
        print(f"  总行数: {len(combined_df)}")
        print(f"  列数: {len(combined_df.columns)}")
        print(f"  标签分布: {combined_df['label'].value_counts().to_dict()}")
        
        return combined_df, file_info
    else:
        print("❌ 未找到Testing_set数据")
        return None, []

def analyze_field_distribution(df, field_name, top_n=20):
    """
    分析指定字段的分布情况
    """
    if field_name not in df.columns:
        print(f"❌ 字段 '{field_name}' 不存在")
        print(f"可用字段: {list(df.columns)}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 字段分析: {field_name}")
    print(f"{'='*60}")
    
    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"  数据类型: {df[field_name].dtype}")
    print(f"  非空值数量: {df[field_name].notna().sum()}")
    print(f"  空值数量: {df[field_name].isna().sum()}")
    print(f"  唯一值数量: {df[field_name].nunique()}")
    
    # 按标签分组分析
    print(f"\n🏷️ 按标签分组分析:")
    for label in sorted(df['label'].unique()):
        label_name = "正常流量" if label == 0 else "PCDN流量"
        subset = df[df['label'] == label]
        print(f"\n  {label_name} (标签={label}):")
        print(f"    样本数: {len(subset)}")
        print(f"    唯一值数: {subset[field_name].nunique()}")
        
        if df[field_name].dtype in ['object', 'string']:
            # 字符串类型 - 显示最频繁的值
            value_counts = subset[field_name].value_counts()
            print(f"    最频繁的值 (前{min(top_n, len(value_counts))}个):")
            for i, (value, count) in enumerate(value_counts.head(top_n).items()):
                percentage = (count / len(subset)) * 100
                print(f"      {i+1:2d}. '{value}' - {count}次 ({percentage:.1f}%)")
        else:
            # 数值类型 - 显示统计信息
            print(f"    统计信息:")
            print(f"      最小值: {subset[field_name].min()}")
            print(f"      最大值: {subset[field_name].max()}")
            print(f"      平均值: {subset[field_name].mean():.4f}")
            print(f"      中位数: {subset[field_name].median():.4f}")
            print(f"      标准差: {subset[field_name].std():.4f}")
    
    # 检查数据泄露风险
    print(f"\n🚨 数据泄露风险检查:")
    
    # 1. 检查是否有完全相同的值在训练和测试中
    if df[field_name].dtype in ['object', 'string']:
        # 字符串类型 - 检查值重叠
        app0_values = set(df[df['label'] == 0][field_name].dropna().unique())
        app1_values = set(df[df['label'] == 1][field_name].dropna().unique())
        
        overlap = app0_values & app1_values
        print(f"  值重叠检查:")
        print(f"    APP_0唯一值数: {len(app0_values)}")
        print(f"    APP_1唯一值数: {len(app1_values)}")
        print(f"    重叠值数: {len(overlap)}")
        
        if overlap:
            print(f"    ⚠️ 发现重叠值 (前10个): {list(overlap)[:10]}")
        else:
            print(f"    ✅ 无重叠值")
    
    # 2. 检查字段是否包含明显的分类信息
    suspicious_patterns = []
    if df[field_name].dtype in ['object', 'string']:
        # 检查是否包含IP地址、URL等可能泄露信息的内容
        sample_values = df[field_name].dropna().astype(str).head(100)
        for value in sample_values:
            if any(pattern in str(value).lower() for pattern in ['http', 'www', 'com', 'net', 'org']):
                suspicious_patterns.append('URL模式')
                break
            if '.' in str(value) and str(value).count('.') >= 3:
                suspicious_patterns.append('IP地址模式')
                break
    
    if suspicious_patterns:
        print(f"    ⚠️ 发现可疑模式: {suspicious_patterns}")
    else:
        print(f"    ✅ 未发现明显可疑模式")
    
    # 3. 检查字段分布是否过于简单
    if df[field_name].nunique() <= 2:
        print(f"    ⚠️ 字段只有{df[field_name].nunique()}个唯一值，可能过于简单")
    elif df[field_name].nunique() > len(df) * 0.9:
        print(f"    ⚠️ 字段唯一值过多({df[field_name].nunique()}/{len(df)})，可能存在过拟合风险")
    else:
        print(f"    ✅ 字段唯一值数量合理")

def visualize_field_distribution(df, field_name, max_categories=20):
    """
    可视化字段分布
    """
    if field_name not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'字段分布分析: {field_name}', fontsize=16)
    
    # 1. 整体分布
    ax1 = axes[0, 0]
    if df[field_name].dtype in ['object', 'string']:
        value_counts = df[field_name].value_counts().head(max_categories)
        value_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('整体分布 (前20个值)')
        ax1.tick_params(axis='x', rotation=45)
    else:
        df[field_name].hist(bins=30, ax=ax1)
        ax1.set_title('整体分布')
    ax1.set_ylabel('频次')
    
    # 2. 按标签分组分布
    ax2 = axes[0, 1]
    for label in sorted(df['label'].unique()):
        label_name = "正常流量" if label == 0 else "PCDN流量"
        subset = df[df['label'] == label]
        
        if df[field_name].dtype in ['object', 'string']:
            value_counts = subset[field_name].value_counts().head(max_categories)
            ax2.plot(value_counts.index, value_counts.values, 'o-', label=label_name, alpha=0.7)
        else:
            ax2.hist(subset[field_name].dropna(), alpha=0.7, label=label_name, bins=20)
    
    ax2.set_title('按标签分组分布')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 标签分布饼图
    ax3 = axes[1, 0]
    label_counts = df['label'].value_counts()
    labels = ['正常流量' if x == 0 else 'PCDN流量' for x in label_counts.index]
    ax3.pie(label_counts.values, labels=labels, autopct='%1.1f%%')
    ax3.set_title('标签分布')
    
    # 4. 字段值分布饼图 (仅对分类字段)
    ax4 = axes[1, 1]
    if df[field_name].dtype in ['object', 'string'] and df[field_name].nunique() <= 10:
        value_counts = df[field_name].value_counts()
        ax4.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        ax4.set_title(f'{field_name} 值分布')
    else:
        ax4.text(0.5, 0.5, f'字段有{df[field_name].nunique()}个唯一值\n无法显示饼图', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'{field_name} 值分布')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    print("🔍 Testing_set字段分布分析工具")
    print("=" * 60)
    
    # 加载数据
    df, file_info = load_testing_data()
    if df is None:
        return
    
    # 显示可用字段
    print(f"\n📋 可用字段列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 交互式分析
    while True:
        print(f"\n{'='*60}")
        field_name = input("请输入要分析的字段名 (输入 'quit' 退出): ").strip()
        
        if field_name.lower() == 'quit':
            break
        
        if field_name not in df.columns:
            print(f"❌ 字段 '{field_name}' 不存在")
            continue
        
        # 分析字段
        analyze_field_distribution(df, field_name)
        
        # 询问是否可视化
        show_plot = input("\n是否显示可视化图表? (y/n): ").strip().lower()
        if show_plot == 'y':
            visualize_field_distribution(df, field_name)
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
