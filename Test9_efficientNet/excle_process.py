import pandas as pd
import numpy as np


def match_and_save_data(file1_path, file2_path, output_path):
    # 读取第一个Excel文件
    df1 = pd.read_excel(file1_path)
    print("第一个Excel文件的前几行数据:")
    print(df1.head())

    # 读取第二个Excel文件（没有列名，手动指定列名）
    df2 = pd.read_excel(file2_path, header=None, names=['index', 'value1', 'value2'])
    print("\n第二个Excel文件的前几行数据:")
    print(df2.head())

    # 确保两个数据框中的index列类型一致
    df1['index'] = df1['index'].astype(int)
    df2['index'] = df2['index'].astype(int)

    # 找出两个数据框中都存在的index值
    common_indices = set(df1['index']).intersection(set(df2['index']))
    print(f"\n共找到 {len(common_indices)} 个匹配的index")

    # 筛选第一个数据框中index在共同index中的行
    matched_df1 = df1[df1['index'].isin(common_indices)]

    # 筛选第二个数据框中index在共同index中的行
    matched_df2 = df2[df2['index'].isin(common_indices)]

    # 按index排序
    matched_df1 = matched_df1.sort_values('index')
    matched_df2 = matched_df2.sort_values('index')

    # 合并两个数据框
    result_df = pd.merge(matched_df1, matched_df2, on='index', how='inner')

    # 重命名列以更清晰
    result_df.columns = ['实际标签', '临床预测标签', '预测概率', 'index', '值1', '值2']

    # 保存到新的Excel文件
    result_df.to_excel(output_path, index=False)
    print(f"\n匹配结果已保存到: {output_path}")
    print("\n匹配结果的前几行:")
    print(result_df.head())

    return result_df


# 使用示例
if __name__ == '__main__':
    # 文件路径
    file1_path = r"D:\path\to\your\first_file.xlsx"  # 请替换为第一个Excel文件的实际路径
    file2_path = r"D:\path\to\your\second_file.xlsx"  # 请替换为第二个Excel文件的实际路径
    output_path = r"D:\path\to\your\matched_results.xlsx"  # 请替换为您想要保存的输出文件路径

    # 执行匹配和保存
    result = match_and_save_data(file1_path, file2_path, output_path)