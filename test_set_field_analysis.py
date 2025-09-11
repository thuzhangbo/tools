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

def normalize_data_types(df):
    """
    标准化数据类型，解决0和0.0不一致问题
    """
    df_normalized = df.copy()
    
    # 数值字段列表
    numeric_fields = [
        'frame.number', 'frame.time_relative', 'ip.version', 'ip.hdr_len', 
        'ip.dsfield', 'ip.len', 'ip.ttl', 'ip.proto', 'ipv6.plen', 'ipv6.nxt',
        'tcp.srcport', 'tcp.dstport', 'tcp.hdr_len', 'tcp.flags.syn', 
        'tcp.flags.ack', 'tcp.payload', 'udp.srcport', 'udp.dstport', 
        'udp.length', 'pcap_duration', 'srcport', 'dstport', 'ulProtoID',
        'dpi_rule_pkt', 'dpi_packets', 'dpi_bytes', 'flow_uplink_traffic',
        'flow_downlink_traffic', 'sum_pkt_len', 'total_pkts', 'srcport_cls',
        'dstport_cls', 'pkt_len_avg', 'pkt_len_max', 'pkt_len_min',
        'up_pkts', 'up_bytes', 'down_pkts', 'down_bytes', 'up_pkt_ratio',
        'down_pkt_ratio', 'up_byte_ratio', 'down_byte_ratio', 
        'up_down_pkt_ratio', 'up_down_byte_ratio', 'iat_avg', 'iat_max',
        'iat_min', 'avg_speed', 'avg_pkt_speed', 'max_burst'
    ]
    
    # 统一转换数值字段为float64
    for field in numeric_fields:
        if field in df_normalized.columns:
            try:
                # 先转换为数值，然后统一为float64
                df_normalized[field] = pd.to_numeric(df_normalized[field], errors='coerce').astype('float64')
            except Exception as e:
                print(f"⚠️ 字段 {field} 类型转换失败: {e}")
    
    return df_normalized

def check_data_type_consistency(df):
    """
    检查数据类型一致性
    """
    # 关键数值字段
    key_fields = ['tcp.payload', 'tcp.srcport', 'tcp.dstport', 'ip.len', 'udp.length', 
                  'frame.number', 'ip.version', 'pkt_len_avg', 'up_bytes', 'down_bytes']
    
    inconsistent_fields = []
    
    for field in key_fields:
        if field in df.columns:
            app0_data = df[df['label'] == 0][field]
            app1_data = df[df['label'] == 1][field]
            
            if app0_data.dtype != app1_data.dtype:
                inconsistent_fields.append(field)
                print(f"  ❌ {field}: APP_0={app0_data.dtype}, APP_1={app1_data.dtype}")
            else:
                print(f"  ✅ {field}: {app0_data.dtype}")
    
    if not inconsistent_fields:
        print("  🎉 所有关键字段数据类型一致！")
    else:
        print(f"  ⚠️ 发现 {len(inconsistent_fields)} 个不一致字段")
    
    return inconsistent_fields

def load_testing_data():
    """
    加载Testing_set数据并标准化数据类型
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
                    
                    # 标准化数据类型
                    df = normalize_data_types(df)
                    
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
        
        # 检查数据类型一致性
        print(f"\n🔍 数据类型一致性检查:")
        check_data_type_consistency(combined_df)
        
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
    
    # 检查数据类型一致性
    print(f"\n🔍 数据类型一致性检查:")
    app0_data = df[df['label'] == 0][field_name]
    app1_data = df[df['label'] == 1][field_name]
    
    if app0_data.dtype != app1_data.dtype:
        print(f"  ❌ 数据类型不一致: APP_0={app0_data.dtype}, APP_1={app1_data.dtype}")
        print(f"  💡 建议: 使用 normalize_data_types() 函数统一数据类型")
    else:
        print(f"  ✅ 数据类型一致: {app0_data.dtype}")
    
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
    else:
        # 数值类型 - 检查值分布差异
        print(f"  数值分布检查:")
        app0_mean = app0_data.mean()
        app1_mean = app1_data.mean()
        app0_std = app0_data.std()
        app1_std = app1_data.std()
        
        print(f"    APP_0: 均值={app0_mean:.4f}, 标准差={app0_std:.4f}")
        print(f"    APP_1: 均值={app1_mean:.4f}, 标准差={app1_std:.4f}")
        
        # 检查是否有明显的分布差异
        if abs(app0_mean - app1_mean) < 1e-10 and abs(app0_std - app1_std) < 1e-10:
            print(f"    ⚠️ 分布几乎完全相同，可能存在数据泄露")
        else:
            print(f"    ✅ 分布存在合理差异")
    
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

def fix_data_type_issues(df):
    """
    修复数据类型问题
    """
    print("\n🔧 修复数据类型问题")
    print("=" * 50)
    
    # 检查并修复数据类型不一致
    inconsistent_fields = check_data_type_consistency(df)
    
    if inconsistent_fields:
        print(f"\n发现 {len(inconsistent_fields)} 个不一致字段，开始修复...")
        df_fixed = normalize_data_types(df)
        
        # 验证修复效果
        print("\n验证修复效果:")
        check_data_type_consistency(df_fixed)
        
        return df_fixed
    else:
        print("\n✅ 数据类型已一致，无需修复")
        return df

def comprehensive_analysis(df):
    """
    综合分析所有字段
    """
    print("\n📊 综合分析报告")
    print("=" * 60)
    
    # 1. 数据类型统计
    print("\n1. 数据类型统计:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} 个字段")
    
    # 2. 关键字段分析
    print("\n2. 关键字段分析:")
    key_fields = ['tcp.payload', 'tcp.srcport', 'tcp.dstport', 'ip.len', 'udp.length']
    
    for field in key_fields:
        if field in df.columns:
            app0_data = df[df['label'] == 0][field]
            app1_data = df[df['label'] == 1][field]
            
            print(f"\n   {field}:")
            print(f"     数据类型: APP_0={app0_data.dtype}, APP_1={app1_data.dtype}")
            print(f"     值范围: APP_0=[{app0_data.min():.2f}, {app0_data.max():.2f}], APP_1=[{app1_data.min():.2f}, {app1_data.max():.2f}]")
            print(f"     均值: APP_0={app0_data.mean():.4f}, APP_1={app1_data.mean():.4f}")
    
    # 3. 数据泄露风险评估
    print("\n3. 数据泄露风险评估:")
    risk_fields = []
    
    for field in df.columns:
        if field in ['source_file', 'app_category', 'label']:
            continue
            
        app0_values = set(df[df['label'] == 0][field].dropna().unique())
        app1_values = set(df[df['label'] == 1][field].dropna().unique())
        
        if field in ['flow_id', 'dpi_file_name', 'dpi_five_tuple']:
            overlap = app0_values & app1_values
            if overlap:
                risk_fields.append(field)
                print(f"   ⚠️ {field}: 发现 {len(overlap)} 个重叠值")
    
    if not risk_fields:
        print("   ✅ 未发现明显的数据泄露风险")
    else:
        print(f"   🚨 发现 {len(risk_fields)} 个高风险字段: {', '.join(risk_fields)}")

def main():
    """
    主函数
    """
    print("🔍 Testing_set字段分布分析工具 (优化版)")
    print("=" * 60)
    
    # 加载数据
    df, file_info = load_testing_data()
    if df is None:
        return
    
    # 修复数据类型问题
    df = fix_data_type_issues(df)
    
    # 综合分析
    comprehensive_analysis(df)
    
    # 显示可用字段
    print(f"\n📋 可用字段列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 交互式分析
    while True:
        print(f"\n{'='*60}")
        print("选择操作:")
        print("1. 分析特定字段")
        print("2. 重新检查数据类型")
        print("3. 查看综合分析报告")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            field_name = input("请输入要分析的字段名: ").strip()
            
            if field_name not in df.columns:
                print(f"❌ 字段 '{field_name}' 不存在")
                continue
            
            # 分析字段
            analyze_field_distribution(df, field_name)
            
            # 询问是否可视化
            show_plot = input("\n是否显示可视化图表? (y/n): ").strip().lower()
            if show_plot == 'y':
                visualize_field_distribution(df, field_name)
                
        elif choice == '2':
            check_data_type_consistency(df)
            
        elif choice == '3':
            comprehensive_analysis(df)
            
        elif choice == '4':
            break
            
        else:
            print("❌ 无效选择，请重新输入")
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
