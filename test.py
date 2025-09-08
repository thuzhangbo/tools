import os
import pandas as pd
from collections import Counter
import glob

def analyze_csv_columns(dataset_path):
    """
    分析指定路径下所有CSV文件的列数是否一致
    找到和大多数不同的文件名
    
    参数:
    dataset_path (str): 数据集根目录路径
    
    返回:
    dict: 包含分析结果的字典
    """
    
    # 存储文件路径和对应的列数
    file_columns = {}
    
    # 递归查找所有CSV文件
    csv_pattern = os.path.join(dataset_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    
    for file_path in csv_files:
        try:
            # 读取CSV文件的第一行来获取列数
            df = pd.read_csv(file_path, nrows=0)  # 只读取列名，不读取数据
            column_count = len(df.columns)
            
            # 使用相对路径作为文件标识
            relative_path = os.path.relpath(file_path, dataset_path)
            file_columns[relative_path] = column_count
            
            print(f"  {relative_path}: {column_count} 列")
            
        except Exception as e:
            print(f"  错误：无法读取文件 {relative_path}: {e}")
            continue
    
    # 统计列数分布
    column_counts = list(file_columns.values())
    count_distribution = Counter(column_counts)
    
    print(f"\n列数分布统计:")
    for count, frequency in sorted(count_distribution.items()):
        print(f"  {count} 列: {frequency} 个文件")
    
    # 找到最常见的列数（众数）
    most_common_count = count_distribution.most_common(1)[0][0]
    most_common_frequency = count_distribution.most_common(1)[0][1]
    
    print(f"\n最常见的列数: {most_common_count} 列 (共 {most_common_frequency} 个文件)")
    
    # 找到列数异常的文件
    abnormal_files = []
    normal_files = []
    
    for file_path, column_count in file_columns.items():
        if column_count != most_common_count:
            abnormal_files.append((file_path, column_count))
        else:
            normal_files.append((file_path, column_count))
    
    # 打印结果
    if abnormal_files:
        print(f"\n发现 {len(abnormal_files)} 个列数异常的文件:")
        for file_path, column_count in abnormal_files:
            print(f"  {file_path}: {column_count} 列 (期望 {most_common_count} 列)")
    else:
        print(f"\n所有文件的列数都一致 ({most_common_count} 列)")
    
    print(f"\n正常文件 ({len(normal_files)} 个):")
    for file_path, column_count in normal_files:
        print(f"  {file_path}: {column_count} 列")
    
    # 返回分析结果
    result = {
        'total_files': len(file_columns),
        'file_columns': file_columns,
        'column_distribution': dict(count_distribution),
        'most_common_count': most_common_count,
        'most_common_frequency': most_common_frequency,
        'abnormal_files': abnormal_files,
        'normal_files': normal_files,
        'is_consistent': len(abnormal_files) == 0
    }
    
    return result

def detailed_column_analysis(dataset_path):
    """
    详细分析CSV文件的列信息
    
    参数:
    dataset_path (str): 数据集根目录路径
    """
    
    csv_pattern = os.path.join(dataset_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print("详细列信息分析:")
    print("=" * 80)
    
    for file_path in csv_files:
        try:
            relative_path = os.path.relpath(file_path, dataset_path)
            print(f"\n文件: {relative_path}")
            
            # 读取文件信息
            df = pd.read_csv(file_path, nrows=5)  # 读取前5行数据
            print(f"  列数: {len(df.columns)}")
            print(f"  行数: {len(pd.read_csv(file_path))}")
            print(f"  列名: {list(df.columns)[:10]}")  # 显示前10个列名
            if len(df.columns) > 10:
                print(f"  ... (总共 {len(df.columns)} 列)")
            
        except Exception as e:
            print(f"  错误：无法读取文件: {e}")

if __name__ == "__main__":
    # 设置数据集路径
    dataset_path = "pcdn_32_pkts_2class_feature_enhance_v17.4_dataset"
    
    print("开始分析CSV文件列数一致性...")
    print("=" * 60)
    
    # 执行基本分析
    result = analyze_csv_columns(dataset_path)
    
    print("\n" + "=" * 60)
    
    # 执行详细分析
    detailed_column_analysis(dataset_path)
    
    print("\n" + "=" * 60)
    print("分析完成！")
