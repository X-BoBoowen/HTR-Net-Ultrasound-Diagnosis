import numpy as np
import pandas as pd
from tqdm import tqdm


def process_features(data):
    labels = data['label']
    feature_list = []
    problem_features = []

    print("===== 开始特征处理 =====")
    for name in data.files:
        if name == 'label':
            continue

        try:
            feature = data[name]
            print(f"\n▶ 处理特征：{name}")
            print(f"   原始形状：{feature.shape} | 数据类型：{feature.dtype}")

            # 维度处理
            if feature.ndim > 2:
                new_shape = (feature.shape[0], -1)
                feature = feature.reshape(new_shape)
                print(f"   多维展平：{feature.shape}")
            elif feature.ndim == 1:
                feature = feature.reshape(-1, 1)
                print(f"   一维转换：{feature.shape}")

            # 数据类型优化
            feature = feature.astype(np.float32)

            # 样本数校验
            if feature_list and (feature.shape[0] != feature_list[0].shape[0]):
                raise ValueError(f"样本数不一致！期望 {feature_list[0].shape[0]}，实际 {feature.shape[0]}")

            feature_list.append(feature)
            print("   ✅ 处理成功")

        except Exception as e:
            problem_features.append((name, str(e)))
            print(f"   ❌ 处理失败：{str(e)}")
            continue

    # 异常报告
    if problem_features:
        print("\n===== 错误汇总 =====")
        for name, err in problem_features:
            print(f"特征名：{name}\n错误信息：{err}\n")
        raise RuntimeError("存在无法处理的特征数据")

    return np.hstack(feature_list), labels


# 主程序
try:
    data = np.load(r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-03\51种故障的数据.npz')

    # 特征处理
    X, labels = process_features(data)
    print(f"\n特征矩阵形状：{X.shape}")

    # 创建DataFrame
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    df['label'] = labels

    # 分块保存
    output_path = '故障数据_带标签.xlsx'
    chunk_size = 5000
    print("\n===== 开始保存数据 =====")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for start in tqdm(range(0, len(df), chunk_size),
                          desc="保存进度",
                          unit="chunk"):
            end = start + chunk_size
            chunk = df.iloc[start:end]
            chunk.to_excel(writer,
                           sheet_name='Data',
                           startrow=start,
                           index=False,
                           header=(start == 0))

    print(f"\n数据已保存至：{output_path}")

except Exception as e:
    print(f"\n❌ 程序运行异常：{str(e)}")