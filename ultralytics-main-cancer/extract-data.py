import pandas as pd

# 读取CSV文件
csv_file_path = 'D:/waibao/yolo11/ultralytics-main/runs/segment/train15/results.csv'  # 将这里的路径替换为你的CSV文件的实际路径
df = pd.read_csv(csv_file_path)

# 提取'metrics/mAP50(B)'列的数据
mAP50_B_data = df[['metrics/mAP50(B)']]

# 将提取的数据保存到新的Excel文件中
excel_file_path = 'D:/waibao/yolo11/ultralytics-main/runs/segment/train15//output.xlsx'  # 指定输出的Excel文件路径
mAP50_B_data.to_excel(excel_file_path, index=False, engine='openpyxl')

print(f"Data from 'metrics/mAP50(B)' column has been saved to {excel_file_path}")