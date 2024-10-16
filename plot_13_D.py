import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from sklearn.linear_model import LinearRegression

def process_csv(input_cell_ruffling, input_sal_per_ruffle, output_file):
    cell_ruffling = pd.read_csv(input_cell_ruffling)
    sal_per_ruffle = pd.read_csv(input_sal_per_ruffle)

    # 初始化映射字典
    cell_ruffling_ratio = {}
    cell_ruffing_std_dev = {}
    sal_per_ruffle_number = {}
    sal_per_ruffle_std_dev = {}
    
    # 遍历cell_ruffling的每一列，建立映射
    for col in cell_ruffling.columns:
        key = col.split('_')[0]  # 获取列名中下划线前的第一个数字作为key
        percentage = cell_ruffling.iloc[3][col]  # 获取第四行的值（索引为3）
        cell_ruffling_ratio[key] = percentage /100
        std_dev = cell_ruffling.iloc[4][col]
        cell_ruffing_std_dev[key] = std_dev / 100

    
    # 遍历sal_per_ruffle的每一列，建立映射
    for col in sal_per_ruffle.columns:
        key = col.split('_')[0]  # 获取列名中下划线前的第一个数字作为key
        value = sal_per_ruffle.iloc[3][col]  # 获取第四行的值（索引为3）
        sal_per_ruffle_number[key] = value
        std_dev = sal_per_ruffle.iloc[4][col]
        sal_per_ruffle_std_dev[key] = std_dev

    # 将字典的key作为列名
    keys = list(cell_ruffling_ratio.keys())
    
    # 创建DataFrame以保存结果
    result_df = pd.DataFrame(columns=keys)
    
    # 添加第一行（cell_ruffling_map的值）
    result_df.loc[0] = [cell_ruffling_ratio[key] for key in keys]

    result_df.loc[1] = [cell_ruffing_std_dev[key] for key in keys]
    
    # 添加第二行（sal_per_ruffle_map的值）
    result_df.loc[2] = [sal_per_ruffle_number[key] for key in keys]

    result_df.loc[3] = [sal_per_ruffle_std_dev[key] for key in keys]    
    
    # 添加第三行（第一行和第二行的乘积）
    result_df.loc[4] = result_df.loc[0] * result_df.loc[2]

    # 计算标准方差的传播公式，并添加到第五行
    propagated_std_dev = []
    for key in keys:
        a = cell_ruffling_ratio[key]
        b = sal_per_ruffle_number[key]
        sigma_a = cell_ruffing_std_dev[key]
        sigma_b = sal_per_ruffle_std_dev[key]
        
        # 计算乘积的标准方差
        sigma_c = np.sqrt((b * sigma_a) ** 2 + (a * sigma_b) ** 2)
        propagated_std_dev.append(sigma_c)
    
    result_df.loc[5] = propagated_std_dev
    
    # 将结果写入CSV文件
    result_df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")

    # 数据准备
    numeric_keys = [float(key) for key in keys]
    # print(numeric_keys)
    y_values = result_df.loc[4].values
    yerr = result_df.loc[5]
    
    # 线性拟合
    nb_point_train = 5

    if len(numeric_keys) >= 2:
        coeffs, cov_matrix = np.polyfit(numeric_keys[:nb_point_train], y_values[:nb_point_train], 1, cov=True)  # 线性拟合（多项式次数1）
        poly = np.poly1d(coeffs)
    
        # 预测
        x_extrapolate = np.array(numeric_keys)
        y_extrapolate = poly(x_extrapolate)
        
        # 计算拟合参数的标准误差
        p_std_err = np.sqrt(np.diag(cov_matrix))

        # 计算外推点的标准误差
        x_mean = np.mean(numeric_keys)
        n = len(numeric_keys)
        yerr_extrapolate = np.sqrt(p_std_err[0]**2 * (x_extrapolate - x_mean)**2 + p_std_err[1]**2 + (p_std_err[0]**2 / n))

        # print('predicted_y_values: ',predicted_y_values)
    else:
        print("数据点不足，无法进行线性拟合。")
        return
    
    #Misselwitz
    # y_origin = [0,0,0.02,0.1,0.6,1.5,3.8,6.8]
    # yerr_origin = [0,0,0,0,0.4,0.5,1.2,1.8]

    origin_number= np.array([0.88, 1.08, 1.64, 2, 3.03, 4.84, 8.4, 13.4])
    origin_percentage = np.array([2, 5, 8, 16, 31, 52, 75, 88])/100
    y_origin = origin_number* (origin_percentage )
    # print('y_origin: ',y_origin)
    # propagation of uncertainty

    std_dev_outside=np.array([0, 0, 0.13, 0.2, 0, 0.27, 0.67, 1])
    std_dev_inside=np.array([0, 0, 0.2, 0, 0.43, 0.47, 0.8, 0.93])
    # 计算方差
    variance_A = std_dev_outside ** 2
    variance_B = std_dev_inside ** 2

    # 方差相加
    total_variance = variance_A + variance_B

    # 计算新的标准差
    sigma_number = np.sqrt(total_variance)
    sigma_percentage = [0, 0, 0, 0.05, 0.01, 0.04, 0.06, 0.02] #[0, 0, 0, 5, 10, 4, 6, 2]
    yerr_origin = np.sqrt((origin_percentage * sigma_number) ** 2 + (origin_number * sigma_percentage) ** 2)


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

    # Misselwitz
    plt.errorbar(keys, y_origin, yerr = yerr_origin,fmt='-o',color = 'red',label='Experimental', capsize= 5 )

    # Simulation
    plt.errorbar(keys, result_df.loc[4], yerr = yerr, fmt='-o', color='black', label='Simulation', capsize=5)
    
    # Extrapolation
    plt.errorbar(keys, y_extrapolate,yerr = yerr_extrapolate, fmt = '--o', color='gray', label='Simulation, extrapolated from low MOI')  # 预测数据
    
    

    plt.xlabel('m.o.i.')
    plt.ylabel('Number of invaded Salmonella per cell')
    # plt.title("D")
    # plt.title('Line Plot with Error Bars')
    plt.legend()
    plt.grid(False)
    
    # 保存图像到文件
    plt_output_path = os.path.join(os.path.dirname(output_file), f'Fig_4_13_D_{nb_point_train}.png')
    plt.savefig(plt_output_path,format='png')
    print(f"PNG plot saved to {plt_output_path}")
    # plt.show()


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Process CSV files to calculate data with propagated standard deviations.')
    parser.add_argument('folder', type=str, help='The folder containing the input CSV files.')

    # 解析命令行参数
    args = parser.parse_args()
    folder = args.folder

    # 定义输入和输出文件路径
    input_cell_ruffling = os.path.join(folder, 'NrRuffle_counted.csv')
    input_sal_per_ruffle = os.path.join(folder, 'SalRuffle_counted.csv')
    output_file = os.path.join(folder, 'Sal_per_cell.csv')

    # 调用处理函数
    process_csv(input_cell_ruffling, input_sal_per_ruffle, output_file)

if __name__ == "__main__":
    main()