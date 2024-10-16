import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import os
import sys
import re

def process_csv_file(file_path, filename):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # 使用空格作为分隔符
        #header=None, delimiter='\s+', 
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)  # 如果文件无法读取，立即中断程序

    # 确保数据行数足够
    if df.shape[0] == 0:
        print(f"File: {filename} is empty.")
        sys.exit(1)

    # 获取最后一行数据
    last_row = df.iloc[-1, 0]  # 获取最后一行的第一个（也是唯一）列数据
    components = last_row.split()  # 按空格分隔

    NrRuffle = components[7]
    return NrRuffle

def sort_by_numbers(name):
    """提取文件夹名字中的数字并返回一个可用于排序的元组"""
    return tuple(map(int, re.findall(r'\d+', name)))

def process_folder(folder_path):
    """遍历文件夹及其子文件夹,处理所有符合条件的CSV文件。"""
    folder_data = {}
    root_folder_name = os.path.basename(folder_path)

    for root, dirs, files in os.walk(folder_path):
        # 跳过空文件夹
        if not files:
            print(f"Skipping empty folder: {root}")
            continue
        
        # 为当前文件夹创建一个列表
        folder_name = os.path.basename(root)
        if folder_name == root_folder_name:
            continue

        folder_data[folder_name] = []
    
        for filename in files:
            if filename.endswith('.csv') and 'summary' not in filename:
                # found_files = True
            
                file_path = os.path.join(root, filename)
                
                # 处理CSV文件
                try:
                    NrRuffle = process_csv_file(file_path, filename)
                    # 将第8列的值存储在对应文件夹的列表中
                    folder_data[folder_name].append(NrRuffle)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    sys.exit(1)

        # 对文件夹列表的数据进行从大到小排序
        folder_data[folder_name].sort(reverse=True)

        # 输出文件夹名称以及已处理的CSV文件数量
        print(f"{folder_name}: {len(folder_data[folder_name])} CSV files processed.")

    # 按文件夹名字中的数字排序
    sorted_folder_data = dict(sorted(folder_data.items(), key=lambda item: sort_by_numbers(item[0])))

    # # 创建DataFrame
    # df = pd.DataFrame(sorted_folder_data)

    # # 保存DataFrame到CSV文件
    # output_csv_path = os.path.join(folder_path, 'NrRuffle_summary.csv')
    # df.to_csv(output_csv_path, index=False)

    # print(f"Data saved to {output_csv_path}")

    return sorted_folder_data


def process_plot_data(folder_data, input_folder):
    """
    处理从process_folder函数返回的数据,生成统计信息并绘制图表。
    
    参数:
    - folder_data: 来自process_folder返回的字典,键为文件夹名,值为该文件夹的CSV数据列表。
    """
    # # 读取CSV文件
    # df = pd.read_csv(input_file)
    # folder_data 是字典，首先将其转为 DataFrame
    folder_data_df = pd.DataFrame(folder_data)

    # 获取所有列的数据
    all_columns = []
    for folder, data_list in folder_data.items():
        all_columns.extend(data_list)
    # 获取总行数

    total_rows = len(folder_data_df)
    # print(total_rows)

    # 创建一个空的DataFrame用于存储统计信息
    stats_df = pd.DataFrame(columns=folder_data_df.columns)

    # 创建一个字典用于存储具有相同前两个数字的列的索引
    prefix_groups = defaultdict(list)
    average_ratios = {}
    std_dev_ratios = {}

    # 遍历每一列
    for col in folder_data_df.columns:
        # 确保数据为数值类型，无法转换的值将会变为 NaN
        folder_data_df[col] = pd.to_numeric(folder_data_df[col], errors='coerce')
        
        # 从第二行开始（跳过第一行），统计非0元素的个数
        non_zero_count = (folder_data_df[col][0:] != 0).sum()
        # print(non_zero_count)

        # 计算非0元素的占比，保留两位小数
        non_zero_ratio = round(non_zero_count / total_rows, 4)

        # 在新的DataFrame中存储统计信息
        stats_df.at[0, col] = total_rows  # 总行数
        stats_df.at[1, col] = non_zero_count  # 非0元素的个数
        stats_df.at[2, col] = non_zero_ratio  # 非0元素的占比

        # 提取列名的第一个数字作为前缀
        prefix = "_".join(col.split('_')[0])
        prefix_groups[prefix].append(col)

    # 计算相同前两个数字的列的比例平均数
    for prefix, cols in prefix_groups.items():
        if len(cols) > 1:
            # 计算所有相关列第三行（比例）的平均值
            avg_ratio = round(stats_df.loc[2, cols].mean()*100, 4)
            std_dev_ratio = round(stats_df.loc[2, cols].std() * 100, 4)
            # 插入到相关列的第四行
            for col in cols:
                stats_df.at[3, col] = avg_ratio
                stats_df.at[4, col] = std_dev_ratio
            average_ratios[prefix] = avg_ratio
            std_dev_ratios[prefix] = std_dev_ratio

    # 将统计信息插入到原始数据的前面
    result_df = pd.concat([stats_df, folder_data_df]).reset_index(drop=True)

    # 保存处理后的CSV文件
    output_csv_path = os.path.join(input_folder, 'NrRuffle_counted.csv')
    result_df.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")

    # 绘制折线图
    plt.figure(figsize=(10, 8))

    # 设置全局字体大小和线条粗细
    plt.rcParams.update({
    'font.size': 18,         # 全局字体大小
    'axes.labelsize': 22,    # 坐标轴标签字体大小
    'axes.titlesize': 20,    # 标题字体大小
    'axes.titleweight': 'bold',
    'lines.linewidth': 3,    # 折线图线条粗细
    'lines.markersize': 8,  # 数据点的大小
    'errorbar.capsize': 5,  # 误差条帽子的大小
    'legend.fontsize': 16,    # 图例字体大小
    })

    # 提取前缀和对应的平均值和标准偏差
    prefixes = list(average_ratios.keys())
    prefixes_str = [item.replace('_', '') for item in prefixes]
    # print(prefixes)
    # print(prefixes_str)

    avg_values = list(average_ratios.values())
    std_dev_values = [std_dev_ratios.get(prefix, 0) for prefix in prefixes]

    if len(prefixes_str) == 8:
        origin_y = [2, 5, 8, 16, 31, 52, 75, 88]
        origin_std_dev = np.array([0, 0, 0, 5, 10, 4, 6, 2])/2
        # plt.title("A")
    elif len(prefixes_str) == 11:
        origin_y =[3,4,10,18,32,52,77,84,93,94,91]
        origin_std_dev =np.array([0, 0, 8, 14, 18, 22, 22, 12, 4, 6, 12])/2
        # plt.title("B")
    
    # 绘制平均数的折线图
    # line_E, =
    plt.errorbar(prefixes_str,origin_y,yerr = origin_std_dev, fmt='-o', color='red', label='Experimental', capsize=5)
    # print('origin_std_dev: ', origin_std_dev)
    # line_S, =
    plt.errorbar(prefixes_str, avg_values, yerr=std_dev_values, fmt='-o', color='black', label='Simulation', capsize=5)
   
    # 设置x轴和y轴标签
    plt.xlabel("m.o.i.")
    plt.ylabel("% cells ruffling")
    

    # 显示图表
    plt.grid(False)

    plt.legend()
    # plt.legend([line_S, line_E], ['Simulation', 'Experimental'])

    if len(prefixes_str) == 8:
        plt_output_path = os.path.join(input_folder, 'Figure_4_13_A.png')
    elif len(prefixes_str) == 11:
        plt_output_path = os.path.join(input_folder, 'Figure_4_13_B.png')
    plt.savefig(plt_output_path, format='png')
    print(f"PNG plot saved to {plt_output_path}")

    # plt.show()


if __name__ == "__main__":
     # 命令行参数解析
    parser = argparse.ArgumentParser(description="Process a folder and generate CSV and graphs.")
    parser.add_argument("folder_path", help="Path to the main folder containing subfolders and CSV files.")
    
    args = parser.parse_args()

    # 处理主文件夹
    folder_data = process_folder(args.folder_path)
    
    # 生成CSV文件并绘制图表
    process_plot_data(folder_data, args.folder_path)
