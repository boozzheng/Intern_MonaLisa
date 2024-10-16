import os
import pandas as pd
import sys
import argparse
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random
import csv
import numpy as np
import seaborn as sns

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

    SalVac = int(components[12])
    SalCyt = int(components[11])

    if SalCyt >= 1:
        NrOfSal = SalCyt + SalVac
        PerOfCyt = SalCyt/NrOfSal * 100

    else:
        NrOfSal = 0 
        PerOfCyt = 0
    return NrOfSal, SalCyt, PerOfCyt

def sort_by_numbers(name):
    """提取文件夹名字中的数字并返回一个可用于排序的元组"""
    return tuple(map(int, re.findall(r'\d+', name)))

def process_folder(folder_path):
    """遍历文件夹及其子文件夹，处理所有符合条件的CSV文件。"""
    folder_data = {}
    root_folder_name = os.path.basename(folder_path)

    for root, dirs, files in os.walk(folder_path):
        
        # 为当前文件夹创建一个列表
        folder_name = os.path.basename(root)
        if folder_name == root_folder_name:
            continue
        
        # 跳过空文件夹
        if not files:
            print(f"Skipping empty folder: {root}")
            continue

        folder_data[folder_name] = []
    
        for filename in files:
            if filename.endswith('.csv') and 'summary' not in filename:
                # found_files = True
            
                file_path = os.path.join(root, filename)
                
                # 处理CSV文件
                try:
                    SalVac_SalCyt,SalCyt,PerOfCyt = process_csv_file(file_path, filename)
                    # 将第8列的值存储在对应文件夹的列表中
                    folder_data[folder_name].append((SalVac_SalCyt, SalCyt, PerOfCyt))
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    sys.exit(1)

        # # 对文件夹列表的数据进行从大到小排序
        # folder_data[folder_name].sort() #reverse=True

        # # 输出文件夹名称以及已处理的CSV文件数量
        # print(f"{folder_name}: {len(folder_data[folder_name])} CSV files processed.")
    
    # Flatten folder_data values into a single list
    all_values = [value for sublist in folder_data.values() for value in sublist]
    # print(len(all_values))
    

    # non_zero_values = [value for value in all_values if value > 0]
    non_zero_values = [(value[0], value[1],value[2]) for value in all_values if value[0] > 0]
    # print(len(non_zero_values))

    """use random process"""
    # if len(non_zero_values) < 201:
    #     print(f"Not enough non-zero values. Found {len(non_zero_values)} values. Exiting.")
    #     sys.exit(1)

    # Randomly select 201 values from non-zero values
    # random.seed()  # Initialize random seed for reproducibility
    # selected_values = random.sample(non_zero_values, 201)
    # selected_values.sort()
    # random_seed = random.getstate()[1][0]  # Get the seed for reproducibility
    # output_csv_path = os.path.join(folder_path, f'SalVac_SalCyt_201.csv')

    """use all cells"""
    selected_values = non_zero_values
    selected_values.sort()
    output_csv_path = os.path.join(folder_path, f'SalVac_SalCyt_{len(selected_values)}.csv')

    with open(output_csv_path, 'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['Seed'])
        # writer.writerow([random_seed])
        writer.writerow(['Sum','SalCyt','PerOfCyt'])
        for value in selected_values:
            writer.writerow([value[0], value[1],value[2]])#([value]) 

    # print(f"Random seed used: {random_seed}")
    print(f"Selected {len(selected_values)} values. Data saved to {output_csv_path}")

    return selected_values
    
# main_folder_path = '/home/bo/StochasticSimulation/Output_6'    
# /home/bo/StochasticSimulation/Output_5
# process_folder(main_folder_path)

def process_data(selected_values):
    """对每个文件夹列表中的数值进行统计，分组并统计数量。"""
   
    grouped_counts = defaultdict(int)
    PerOfCyt_groups = defaultdict(list)

    total_count = len(selected_values)

    for value_tuple in selected_values:
        SalVac_SalCyt = value_tuple[0]  # 选择第一个值 SalVac_SalCyt 进行分组
        PerOfCyt = value_tuple[2]

        if SalVac_SalCyt > 10:
            grouped_counts['>10'] += 1
        else:
            grouped_counts[SalVac_SalCyt] += 1

        if SalVac_SalCyt>=1 and SalVac_SalCyt<=5:
            PerOfCyt_groups['1-5'].append(PerOfCyt)
        elif SalVac_SalCyt >= 6 and SalVac_SalCyt <= 10:
            PerOfCyt_groups['6-10'].append(PerOfCyt)
        elif SalVac_SalCyt > 10:
            PerOfCyt_groups['>10'].append(PerOfCyt)

    # 计算百分比
    grouped_percentages = {k: (v / total_count) * 100 for k, v in grouped_counts.items()}

    # print(f"Value percentages: {grouped_percentages}")
    # print("-" * 40)
    # print(type(PerOfCyt_groups))

    return grouped_percentages, PerOfCyt_groups



def plot_grouped_percentages(selected_values,grouped_percentages,output_dir):#
    """根据分组计数绘制柱状图"""
    labels = list(map(str, grouped_percentages.keys()))  # 将labels转换为字符串类型
    percentages = list(grouped_percentages.values())
   

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
    'font.size': 18,         # 全局字体大小
    'axes.labelsize': 24,    # 坐标轴标签字体大小
    'axes.labelweight':'bold',
    'axes.titlesize': 24,    # 标题字体大小
    'axes.titleweight': 'bold',
    'lines.linewidth': 3,    # 折线图线条粗细
    'lines.markersize': 8,  # 数据点的大小
    'errorbar.capsize': 5,  # 误差条帽子的大小
    'legend.fontsize': 16,    # 图例字体大小
    })
    plt.bar(labels, percentages, color='slateblue')
    plt.xlabel('Number of bacteria/cell')
    plt.ylabel('Percentage of scored cells')
    # plt.title('A')
    # plt.title(f'Value Distribution in {folder_name}')

    # 生成保存文件的路径和文件名
    """use random process"""
    # output_file = os.path.join(output_dir, f"Fig_4_14_A_201.png")
    """use all cells"""
    output_file = os.path.join(output_dir, f"Fig_4_14_A_{len(selected_values)}.png")
    
    # 保存图表为PNG文件
    plt.savefig(output_file)

    # plt.show()

    print(f"Histogram plot saved as {output_file}")

# def process_PerOfCyt_groups(PerOfCyt_groups):
    # """处理 PerOfCyt_groups，检测重复的值，并为其添加序列号。"""
    # processed_PerOfCyt_groups = defaultdict(list)
    
    # for group, values in PerOfCyt_groups.items():
    #     # 统计每个值的出现次数
    #     value_counts = defaultdict(int)
    #     for value in values:
    #         value_counts[value] += 1
        
    #     # 生成序列号
    #     seen_values = defaultdict(int)
    #     for value in values:
    #         count = seen_values[value]
    #         processed_PerOfCyt_groups[group].append((value, count))
    #         seen_values[value] += 1
    
    # return processed_PerOfCyt_groups


# def plot_grouped_perofcyt(selected_values,processed_PerOfCyt_groups, output_dir):
#     plt.figure(figsize=(10, 6))
    
#     # 定义组的顺序映射
#     group_order = {'1-5': 1, '6-10': 2, '>10': 3}

#     for group, values_with_seq in processed_PerOfCyt_groups.items():
#         # 分开值和序列号
#         values, seqs = zip(*values_with_seq)
        
#         # 生成位置索引并添加偏移量
#         x_positions = np.array([group_order[group]]*len(values)) + np.array(seqs) * 0.1
        
#         plt.scatter(x_positions, values, label=group, 
#                      alpha=0.7)
#     #marker=group_markers[group], color=group_colors[group],
    
#     plt.xlabel('Groups')
#     plt.ylabel('Percentage of Cytoplasm (%)')
#     plt.title('Scatter Plot of PerOfCyt by Groups')
#     plt.legend(title='Groups')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # 保存图形
#     output_file = os.path.join(output_dir, f'Fig_4_14_B_{len(selected_values)}.png')
#     plt.savefig(output_file)
#     plt.show()


def prepare_data_for_swarmplot(PerOfCyt_groups):
    """将 PerOfCyt_groups 转换为适合 seaborn swarmplot 的 DataFrame 格式。"""
    data = []
    
    for group, values in PerOfCyt_groups.items():
        for value in values:
            data.append({'Group': group, 'PerOfCyt': value})
    
    df = pd.DataFrame(data)
    return df

def plot_swarmplot(selected_values,df, output_dir):
    """使用 seaborn 绘制 swarmplot 并保存为 PNG 文件。"""
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
    'font.size': 18,         # 全局字体大小
    'axes.labelsize': 24,    # 坐标轴标签字体大小
    'axes.labelweight':'bold',
    'axes.titlesize': 24,    # 标题字体大小
    'axes.titleweight': 'bold',
    'lines.linewidth': 3,    # 折线图线条粗细
    'lines.markersize': 8,  # 数据点的大小
    'errorbar.capsize': 5,  # 误差条帽子的大小
    'legend.fontsize': 16,    # 图例字体大小
    })
    sns.swarmplot(x='Group', y='PerOfCyt', data=df,color='green')#, palette='viridis'
    # plt.title('Swarmplot of PerOfCyt by SalVac_SalCyt Range')
    plt.xlabel('Number of bacteria/cell')
    plt.ylabel(f'Cytosolic bacteria (% of total)')
    # plt.title('B')
    output_path = os.path.join(output_dir, f'Fig_4_14_B_{len(selected_values)}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Swarmplot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of CSV files and generate analysis plots.")
    parser.add_argument("folder", help="Path to the folder containing the CSV files.")
    args = parser.parse_args()

    main_folder_path = args.folder
    # main_folder_path = '/home/bo/StochasticSimulation/Output_6'
    output_dir = main_folder_path

    # 处理文件夹并获取选择出的201个数值
    selected_values = process_folder(main_folder_path)  
    grouped_percentages,PerOfCyt_groups = process_data(selected_values)  # 统计数据
    # processed_PerOfCyt_groups = process_PerOfCyt_groups(PerOfCyt_groups)
    df = prepare_data_for_swarmplot(PerOfCyt_groups)
    # 生成并保存柱状图
    plot_grouped_percentages(selected_values, grouped_percentages, output_dir)#
    # plot_grouped_perofcyt(selected_values,processed_PerOfCyt_groups,output_dir)
    plot_swarmplot(selected_values,df,output_dir)


