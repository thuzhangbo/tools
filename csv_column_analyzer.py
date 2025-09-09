#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCDN流量分类CSV列分析脚本
分析Training_set、Validation_set、Testing_set中CSV文件的列结构
"""

import os
import pandas as pd
from collections import defaultdict
import json

def get_all_csv_files(base_path, dataset_type):
    """
    获取指定数据集类型下所有CSV文件的路径
    
    Args:
        base_path: 数据集根目录路径
        dataset_type: 数据集类型 ('Training_set', 'Validation_set', 'Testing_set')
    
    Returns:
        dict: {文件路径: 所属类别}
    """
    csv_files = {}
    dataset_path = os.path.join(base_path, dataset_type)
    
    if not os.path.exists(dataset_path):
        print(f"警告: 路径 {dataset_path} 不存在")
        return csv_files
    
    # 遍历APP_0和APP_1目录
    for app_dir in ['APP_0', 'APP_1']:
        app_path = os.path.join(dataset_path, app_dir)
        if os.path.exists(app_path):
            for file in os.listdir(app_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(app_path, file)
                    csv_files[file_path] = app_dir
                    
    return csv_files

def analyze_csv_columns(csv_files):
    """
    分析CSV文件的列结构
    
    Args:
        csv_files: dict，{文件路径: 所属类别}
    
    Returns:
        tuple: (所有列名集合的字典, 共有列名集合, 文件列信息字典)
    """
    file_columns = {}
    all_columns_sets = []
    
    print(f"\n正在分析 {len(csv_files)} 个CSV文件...")
    
    for file_path, category in csv_files.items():
        try:
            # 只读取第一行来获取列名
            df = pd.read_csv(file_path, nrows=0)
            columns = set(df.columns.tolist())
            file_columns[file_path] = {
                'columns': columns,
                'column_count': len(columns),
                'category': category,
                'filename': os.path.basename(file_path)
            }
            all_columns_sets.append(columns)
            print(f"  ✓ {os.path.basename(file_path)} ({category}): {len(columns)} 列")
            
        except Exception as e:
            print(f"  ✗ 读取文件失败 {file_path}: {e}")
            continue
    
    # 计算所有文件的共有列
    if all_columns_sets:
        common_columns = set.intersection(*all_columns_sets)
    else:
        common_columns = set()
    
    return file_columns, common_columns, all_columns_sets

def find_unique_columns(file_columns, common_columns):
    """
    找到拥有独特列名的CSV文件
    
    Args:
        file_columns: 文件列信息字典
        common_columns: 共有列名集合
    
    Returns:
        dict: 拥有独特列的文件信息
    """
    unique_files = {}
    
    for file_path, info in file_columns.items():
        unique_cols = info['columns'] - common_columns
        if unique_cols:
            unique_files[file_path] = {
                'filename': info['filename'],
                'category': info['category'],
                'unique_columns': unique_cols,
                'unique_count': len(unique_cols)
            }
    
    return unique_files

def print_analysis_results(dataset_name, file_columns, common_columns, unique_files):
    """
    打印分析结果
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} 分析结果")
    print(f"{'='*60}")
    
    # 基本统计
    total_files = len(file_columns)
    app0_files = len([f for f in file_columns.values() if f['category'] == 'APP_0'])
    app1_files = len([f for f in file_columns.values() if f['category'] == 'APP_1'])
    
    print(f"文件统计:")
    print(f"  总文件数: {total_files}")
    print(f"  APP_0 (正常流量): {app0_files} 个文件")
    print(f"  APP_1 (PCDN流量): {app1_files} 个文件")
    
    # 列统计
    all_column_counts = [info['column_count'] for info in file_columns.values()]
    min_cols = min(all_column_counts) if all_column_counts else 0
    max_cols = max(all_column_counts) if all_column_counts else 0
    
    print(f"\n列数统计:")
    print(f"  最少列数: {min_cols}")
    print(f"  最多列数: {max_cols}")
    print(f"  共有列数: {len(common_columns)}")
    
    # 共有列前10个示例
    print(f"\n共有列示例 (前10个):")
    common_list = sorted(list(common_columns))
    for i, col in enumerate(common_list[:10]):
        print(f"  {i+1}. {col}")
    if len(common_list) > 10:
        print(f"  ... 还有 {len(common_list) - 10} 个列")
    
    # 独特列信息
    print(f"\n拥有独特列的文件:")
    if unique_files:
        for file_path, info in unique_files.items():
            print(f"  文件: {info['filename']} ({info['category']})")
            print(f"    独特列数: {info['unique_count']}")
            print(f"    独特列: {', '.join(sorted(list(info['unique_columns'])))}")
    else:
        print("  无文件拥有独特列（所有文件列结构完全一致）")

def main():
    """
    主函数
    """
    print("PCDN流量分类数据集CSV列分析工具")
    print("=" * 60)
    
    # 数据集基础路径
    base_path = "pcdn_32_pkts_2class_feature_enhance_v17.4_dataset"
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 数据集路径 {base_path} 不存在")
        return
    
    # 存储所有分析结果
    results = {}
    
    # 分析三个数据集
    datasets = ['Training_set', 'Validation_set', 'Testing_set']
    
    for dataset in datasets:
        print(f"\n正在分析 {dataset}...")
        
        # 获取CSV文件列表
        csv_files = get_all_csv_files(base_path, dataset)
        
        if not csv_files:
            print(f"  警告: 在 {dataset} 中未找到CSV文件")
            results[dataset] = {
                'file_columns': {},
                'common_columns': set(),
                'unique_files': {},
                'file_count': 0
            }
            continue
        
        # 分析CSV列结构
        file_columns, common_columns, all_columns_sets = analyze_csv_columns(csv_files)
        
        # 找到独特列
        unique_files = find_unique_columns(file_columns, common_columns)
        
        # 存储结果
        results[dataset] = {
            'file_columns': file_columns,
            'common_columns': common_columns,
            'unique_files': unique_files,
            'file_count': len(csv_files)
        }
        
        # 打印结果
        print_analysis_results(dataset, file_columns, common_columns, unique_files)
    
    # 计算三个数据集的交集
    print(f"\n{'='*60}")
    print("三个数据集共有列的交集分析")
    print(f"{'='*60}")
    
    # 获取每个数据集的共有列
    train_common = results.get('Training_set', {}).get('common_columns', set())
    val_common = results.get('Validation_set', {}).get('common_columns', set())
    test_common = results.get('Testing_set', {}).get('common_columns', set())
    
    # 计算三个数据集的交集
    if train_common and val_common and test_common:
        intersection = train_common & val_common & test_common
        print(f"\nTrain共有列数: {len(train_common)}")
        print(f"Validation共有列数: {len(val_common)}")
        print(f"Test共有列数: {len(test_common)}")
        print(f"三个数据集的交集列数: {len(intersection)}")
        
        if intersection:
            print(f"\n交集列名 (前20个):")
            intersection_list = sorted(list(intersection))
            for i, col in enumerate(intersection_list[:20]):
                print(f"  {i+1}. {col}")
            if len(intersection_list) > 20:
                print(f"  ... 还有 {len(intersection_list) - 20} 个列")
                
            # 保存交集列名到文件
            with open('common_columns_intersection.txt', 'w', encoding='utf-8') as f:
                f.write("Training_set、Validation_set、Testing_set共有列的交集:\n")
                f.write("=" * 50 + "\n")
                f.write(f"交集列数: {len(intersection)}\n\n")
                for i, col in enumerate(intersection_list, 1):
                    f.write(f"{i}. {col}\n")
            print(f"\n交集列名已保存到: common_columns_intersection.txt")
        else:
            print("\n警告: 三个数据集没有共同的列!")
    else:
        print("\n警告: 某些数据集分析失败，无法计算交集")
    
    # 分析各数据集独有的列
    print(f"\n各数据集独有列分析:")
    if train_common:
        train_only = train_common - (val_common | test_common)
        print(f"  Training_set独有列数: {len(train_only)}")
        if train_only:
            print(f"    独有列: {', '.join(sorted(list(train_only)))}")
    
    if val_common:
        val_only = val_common - (train_common | test_common)
        print(f"  Validation_set独有列数: {len(val_only)}")
        if val_only:
            print(f"    独有列: {', '.join(sorted(list(val_only)))}")
    
    if test_common:
        test_only = test_common - (train_common | val_common)
        print(f"  Testing_set独有列数: {len(test_only)}")
        if test_only:
            print(f"    独有列: {', '.join(sorted(list(test_only)))}")
    
    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
