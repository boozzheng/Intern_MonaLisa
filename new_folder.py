import os
import itertools
import argparse

# 定义命令行参数解析
parser = argparse.ArgumentParser(description="Generate folders based on provided moi, it, and NrOfTimes values.")

# 添加命令行参数
parser.add_argument('--moi', nargs='+', required=True, help='List of MOI values, separated by space.')
parser.add_argument('--it', nargs='+', required=True, help='List of IT values, separated by space.')
parser.add_argument('--times', nargs='+', required=True, help='List of Number of Experiments, separated by space.')

# 解析命令行参数
args = parser.parse_args()

# 定义三个列表
# moi = ["1", "2", "4","8","16","32","64","128",'256','512','1024']
# # moi = ['','','','','','','','','','']
# it = ["20"]
# NrOfTimes = ["1", "2", "3","4"]

# 获取所有可能的组合
combinations = itertools.product(args.moi, args.it, args.times)

# 定义创建文件夹的根目录
root_directory = "./generated_folders"

# 创建根目录（如果它不存在）
os.makedirs(root_directory, exist_ok=True)

# 遍历所有组合并创建相应的文件夹
for combination in combinations:
    folder_name = "_".join(combination)  # 组合名称，使用下划线连接
    folder_path = os.path.join(root_directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")
