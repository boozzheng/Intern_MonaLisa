import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
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

    SalRuffle = components[9]
    return SalRuffle

def sort_by_numbers(name):
    """提取文件夹名字中的数字并返回一个可用于排序的元组"""
    return tuple(map(int, re.findall(r'\d+', name)))

def process_folder(folder_path):
    """遍历文件夹及其子文件夹，处理所有符合条件的CSV文件。"""
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
                    SalRuffle = process_csv_file(file_path, filename)
                    # 将第8列的值存储在对应文件夹的列表中
                    folder_data[folder_name].append(SalRuffle)
                    
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
    # output_csv_path = os.path.join(folder_path, 'SalRuffle_summary.csv') #_1ruffle
    # df.to_csv(output_csv_path, index=False)

    # print(f"Data saved to {output_csv_path}")

    return sorted_folder_data

def process_plot_data(folder_data, input_folder):
    # 读取CSV文件
    # df = pd.read_csv(input_file)
    df = pd.DataFrame(folder_data)

    # 获取总行数
    total_rows = len(df)
    # print(total_rows)

    # 创建一个空的DataFrame用于存储统计信息
    stats_df = pd.DataFrame(columns=df.columns)

    # 创建一个字典用于存储具有相同前两个数字的列的索引
    prefix_groups = defaultdict(list)
    average_means = {}
    std_dev_means = {}

    # 遍历每一列
    for col in df.columns:

        df[col] = pd.to_numeric(df[col], errors='coerce')

        # 统计非0元素的个数
        non_zero_count = (df[col][0:] != 0).sum()
        
        # print(non_zero_count)

        # 计算非0元素的数值总和
        non_zero_sum = df[col][df[col] != 0].sum()

        # 计算非0元素的平均数
        if non_zero_count > 0:
            non_zero_mean = round(non_zero_sum / non_zero_count,2)
        else:
            non_zero_mean = 0  # 如果没有非0元素，平均值为0

        # 在新的DataFrame中存储统计信息
        stats_df.at[0, col] = non_zero_count  # 非0元素的个数
        stats_df.at[1, col] = non_zero_sum
        stats_df.at[2, col] = non_zero_mean
        
        # 提取列名的前两个数字作为前缀
        prefix = "_".join(col.split('_')[0]) #[0:2]# 假设前两个部分作为前缀
        prefix_groups[prefix].append(col)

    # 计算相同前两个数字的列的平均salruffle的平均数
    for prefix, cols in prefix_groups.items():
        if len(cols) > 1:
            # 计算所有相关列第三行（）的平均值
            # avg_mean = round(stats_df.loc[2, cols].mean(), 2)
            

            total_sum = stats_df.loc[1, cols].sum()  # 第二行相加的总和
            total_count = stats_df.loc[0, cols].sum()  # 第一行相加的总和
            
            if total_count > 0:
                avg_mean = round(total_sum / total_count, 2)
            else:
                avg_mean = 0  # 如果总计数为0，平均数为0

        #   计算符合前缀的所有列的所有非0元素的标准偏差
            # all_non_zero_values = []
            # for col in cols:
            #     all_non_zero_values.extend(df[col][df[col] != 0].tolist())

            # print('all_non_zero_values: ',all_non_zero_values)
            
            # if len(all_non_zero_values) > 1:
            #     std_dev = round(pd.Series(all_non_zero_values).std(), 2)
            # else:
            #     std_dev = 0  # 如果没有足够的数据计算标准偏差
             
            std_dev = round(stats_df.loc[2, cols].std(), 2)

            # 插入到相关列的第四行
            for col in cols:
                stats_df.at[3, col] = avg_mean
                stats_df.at[4, col] = std_dev
            average_means[prefix] = avg_mean
            std_dev_means[prefix] = std_dev

    # 将统计信息插入到原始数据的前面
    result_df = pd.concat([stats_df, df]).reset_index(drop=True)

    # 保存处理后的CSV文件
    output_csv_path = os.path.join(input_folder, 'SalRuffle_counted.csv')
    result_df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved as {output_csv_path}")

    # 绘制折线图
    plt.figure(figsize=(10, 8))

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
    prefixes = list(average_means.keys())
    prefixes_str = [item.replace('_', '') for item in prefixes]
    # print(prefixes)
    # print(prefixes_str)

    avg_values = list(average_means.values())
    std_dev_values = [std_dev_means.get(prefix, 0) for prefix in prefixes]
    
    # Bo
    origin_y = [0.88, 1.08, 1.64, 2, 3.03, 4.84, 8.4, 13.4]
    # Bo_inside
    # origin_y =[0.48,0.6,1.04,1.2,1.88,3,5.12,7.72]
    # Jenni
    # origin_y = [1.06, 1.2, 1.7, 1.9, 3.1, 4.8, 8.2, 12.9]
    std_dev_outside=np.array([0, 0, 0.13, 0.2, 0, 0.27, 0.67, 1])
    std_dev_inside=np.array([0, 0, 0.2, 0, 0.43, 0.47, 0.8, 0.93])
    # 计算方差
    variance_A = std_dev_outside ** 2
    variance_B = std_dev_inside ** 2

    # 方差相加
    total_variance = variance_A + variance_B

    # 计算新的标准差
    origin_std_dev = np.sqrt(total_variance)
    # print('origin_std_dev: ',origin_std_dev)
    

    # 绘制平均数的折线图
    plt.errorbar(prefixes_str, origin_y, yerr=origin_std_dev, fmt='-o', color='red', label='Experimental', capsize=5)
    plt.errorbar(prefixes_str, avg_values, yerr=std_dev_values, fmt='-o', color='black', label='Simulation', capsize=5)
    
    # 设置x轴和y轴标签
    plt.xlabel("m.o.i.")
    plt.ylabel("number Salmonella per ruffle")
    # plt.title("C")

    # 设置x轴的刻度标签为纯数字
    # plt.xticks(ticks=prefixes, labels=prefixes)

    # 设置图表标题
    # plt.title("Cell Ruffling vs. MOI")

    # 显示图表
    plt.grid(False)
    plt.legend()
    

    plt_output_path = os.path.join(input_folder, 'Figure_4_13_C.png')
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