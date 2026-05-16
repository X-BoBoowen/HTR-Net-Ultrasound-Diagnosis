# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# def confidence_based_fusion(model1_result, model1_confidence, model2_result, model2_confidence):
#     """
#     基于置信度动态加权融合两个模型的预测结果
#
#     参数:
#     model1_result: 模型1的预测结果 (0或1)
#     model1_confidence: 模型1的置信度
#     model2_result: 模型2的预测结果 (0或1)
#     model2_confidence: 模型2的置信度
#
#     返回:
#     final_prediction: 最终预测的类别 (0或1)
#     fusion_confidence: 融合后的置信度
#     model1_weight: 模型1在融合中的权重
#     model2_weight: 模型2在融合中的权重
#     """
#     # 计算每个模型的置信度
#     # 对于二分类问题，置信度可以直接使用
#     conf1 = model1_confidence
#     conf2 = model2_confidence
#
#     # 计算归一化权重（置信度越高权重越大）
#     total_confidence = conf1 + conf2
#     if total_confidence == 0:
#         # 防止除零
#         model1_weight = model2_weight = 0.5
#     else:
#         model1_weight = conf1 / total_confidence
#         model2_weight = conf2 / total_confidence
#
#     # 加权融合预测结果
#     # 将预测结果转换为概率形式
#     # 如果预测为1，概率为置信度；如果预测为0，概率为1-置信度
#     if model1_result == 1:
#         prob1 = conf1
#     else:
#         prob1 = 1 - conf1
#
#     if model2_result == 1:
#         prob2 = conf2
#     else:
#         prob2 = 1 - conf2
#
#     # 加权融合概率
#     fusion_prob = (prob1 * model1_weight) + (prob2 * model2_weight)
#
#     # 获取最终预测类别
#     final_prediction = 1 if fusion_prob > 0.5 else 0
#
#     # 计算融合后的置信度
#     fusion_confidence = fusion_prob if final_prediction == 1 else 1 - fusion_prob
#
#     return final_prediction, fusion_confidence, model1_weight, model2_weight
#
#
# def apply_fusion_to_dataframe(input_file, output_file):
#     # 读取Excel文件
#     df = pd.read_excel(input_file)
#     print("原始数据前几行:")
#     print(df.head())
#
#     # 重命名列名以更清晰
#     df.columns = ['label', 'model1_result', 'model1_confidence', 'model2_result', 'model2_confidence']
#
#     # 应用融合函数到每一行
#     fusion_results = []
#     for _, row in df.iterrows():
#         final_pred, fusion_conf, w1, w2 = confidence_based_fusion(
#             row['model1_result'], row['model1_confidence'],
#             row['model2_result'], row['model2_confidence']
#         )
#         fusion_results.append({
#             '融合预测标签': final_pred,
#             '融合置信度': fusion_conf,
#             '模型1权重': w1,
#             '模型2权重': w2
#         })
#
#     # 将融合结果添加到DataFrame
#     fusion_df = pd.DataFrame(fusion_results)
#     df = pd.concat([df, fusion_df], axis=1)
#
#     # 计算融合后的准确率
#     fusion_accuracy = accuracy_score(df['label'], df['融合预测标签'])
#     print(f"\n融合模型在整个数据集上的准确率: {fusion_accuracy:.4f}")
#
#     # 显示模型性能比较
#     model1_accuracy = accuracy_score(df['label'], df['model1_result'])
#     model2_accuracy = accuracy_score(df['label'], df['model2_result'])
#
#     print(f"\n模型1准确率: {model1_accuracy:.4f}")
#     print(f"模型2准确率: {model2_accuracy:.4f}")
#     print(f"融合模型准确率: {fusion_accuracy:.4f}")
#
#     # 计算改进程度
#     improvement_vs_model1 = fusion_accuracy - model1_accuracy
#     improvement_vs_model2 = fusion_accuracy - model2_accuracy
#
#     print(f"\n融合模型相比模型1的改进: {improvement_vs_model1:.4f}")
#     print(f"融合模型相比模型2的改进: {improvement_vs_model2:.4f}")
#
#     # 绘制混淆矩阵
#     cm = confusion_matrix(df['label'], df['融合预测标签'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('融合模型混淆矩阵')
#     plt.ylabel('真实标签')
#     plt.xlabel('预测标签')
#     plt.savefig('confusion_matrix_fusion.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print("混淆矩阵已保存为 'confusion_matrix_fusion.png'")
#
#     # 绘制权重分布图
#     plt.figure(figsize=(10, 6))
#     plt.hist(df['模型1权重'], alpha=0.5, label='模型1权重')
#     plt.hist(df['模型2权重'], alpha=0.5, label='模型2权重')
#     plt.title('模型权重分布')
#     plt.xlabel('权重')
#     plt.ylabel('频次')
#     plt.legend()
#     plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print("权重分布图已保存为 'weight_distribution.png'")
#
#     # 保存结果到Excel
#     df.to_excel(output_file, index=False)
#     print(f"\n融合结果已保存到: {output_file}")
#
#     return df
#
#
# # 使用示例
# if __name__ == '__main__':
#     input_file = r"D:\waibao\cancer_detect\predict_data\融合结果.xlsx"  # 请替换为您的输入Excel文件路径
#     output_file = r"D:\waibao\cancer_detect\predict_data\融合结果_weighted.xlsx"  # 请替换为您想要保存的输出文件路径
#
#     # 执行融合
#     result_df = apply_fusion_to_dataframe(input_file, output_file)
#
#     # 显示融合后的前几行结果
#     print("\n融合后的数据前几行:")
#     print(result_df[['label', 'model1_result', 'model1_confidence',
#                      'model2_result', 'model2_confidence',
#                      '融合预测标签', '融合置信度', '模型1权重', '模型2权重']].head(10))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def confidence_based_fusion(model1_result, model1_confidence, model2_result, model2_confidence):
    """
    基于置信度动态加权融合两个模型的预测结果

    参数:
    model1_result: 模型1的预测结果 (0或1)
    model1_confidence: 模型1的置信度
    model2_result: 模型2的预测结果 (0或1)
    model2_confidence: 模型2的置信度

    返回:
    final_prediction: 最终预测的类别 (0或1)
    fusion_confidence: 融合后的置信度
    model1_weight: 模型1在融合中的权重
    model2_weight: 模型2在融合中的权重
    """
    # 计算每个模型的置信度
    # 对于二分类问题，置信度可以直接使用
    conf1 = model1_confidence
    conf2 = model2_confidence

    # 计算归一化权重（置信度越高权重越大）
    total_confidence = conf1 + conf2
    if total_confidence == 0:
        # 防止除零
        model1_weight = model2_weight = 0.5
    else:
        model1_weight = conf1 / total_confidence
        model2_weight = conf2 / total_confidence

    # 加权融合预测结果
    # 将预测结果转换为概率形式
    # 如果预测为1，概率为置信度；如果预测为0，概率为1-置信度
    if model1_result == 1:
        prob1 = conf1
    else:
        prob1 = 1 - conf1

    if model2_result == 1:
        prob2 = conf2
    else:
        prob2 = 1 - conf2

    # 加权融合概率
    fusion_prob = (prob1 * model1_weight) + (prob2 * model2_weight)

    # 获取最终预测类别
    final_prediction = 1 if fusion_prob > 0.5 else 0

    # 计算融合后的置信度
    fusion_confidence = fusion_prob if final_prediction == 1 else 1 - fusion_prob

    return final_prediction, fusion_confidence, model1_weight, model2_weight


def random_forest_fusion(df, test_size=0.2, random_state=42):
    """
    使用随机森林算法融合两个模型的预测结果和置信度

    参数:
    df: 包含两个模型预测结果和置信度的DataFrame
    test_size: 测试集比例
    random_state: 随机种子

    返回:
    rf_accuracy: 随机森林融合的准确率
    rf_predictions: 随机森林的预测结果
    rf_proba: 随机森林的预测概率
    """
    # 准备特征和标签
    X = df[['model1_result', 'model1_confidence', 'model2_result', 'model2_confidence']]
    y = df['label']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train_scaled, y_train)

    # 预测
    rf_predictions = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)

    # 计算准确率
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print(f"\n随机森林融合模型在测试集上的准确率: {rf_accuracy:.4f}")

    # 在整个数据集上进行预测（使用交叉验证或保留一部分数据的方式更合适，这里简化处理）
    X_scaled = scaler.transform(X)
    all_rf_predictions = rf_model.predict(X_scaled)
    all_rf_proba = rf_model.predict_proba(X_scaled)

    # 计算整个数据集上的准确率
    all_rf_accuracy = accuracy_score(y, all_rf_predictions)
    print(f"随机森林融合模型在整个数据集上的准确率: {all_rf_accuracy:.4f}")

    # 绘制特征重要性图
    feature_importance = rf_model.feature_importances_
    features = ['model1_result', 'model1_confidence', 'model2_result', 'model2_confidence']

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance)
    plt.title('随机森林特征重要性')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("特征重要性图已保存为 'rf_feature_importance.png'")

    return all_rf_accuracy, all_rf_predictions, all_rf_proba


def apply_fusion_to_dataframe(input_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)
    print("原始数据前几行:")
    print(df.head())

    # 重命名列名以更清晰
    df.columns = ['label', 'model1_result', 'model1_confidence', 'model2_result', 'model2_confidence']

    # 应用融合函数到每一行
    fusion_results = []
    for _, row in df.iterrows():
        final_pred, fusion_conf, w1, w2 = confidence_based_fusion(
            row['model1_result'], row['model1_confidence'],
            row['model2_result'], row['model2_confidence']
        )
        fusion_results.append({
            '融合预测标签': final_pred,
            '融合置信度': fusion_conf,
            '模型1权重': w1,
            '模型2权重': w2
        })

    # 将融合结果添加到DataFrame
    fusion_df = pd.DataFrame(fusion_results)
    df = pd.concat([df, fusion_df], axis=1)

    # 计算融合后的准确率
    fusion_accuracy = accuracy_score(df['label'], df['融合预测标签'])
    print(f"\n置信度融合模型在整个数据集上的准确率: {fusion_accuracy:.4f}")

    # 显示模型性能比较
    model1_accuracy = accuracy_score(df['label'], df['model1_result'])
    model2_accuracy = accuracy_score(df['label'], df['model2_result'])

    print(f"\n模型1准确率: {model1_accuracy:.4f}")
    print(f"模型2准确率: {model2_accuracy:.4f}")
    print(f"置信度融合模型准确率: {fusion_accuracy:.4f}")

    # 计算改进程度
    improvement_vs_model1 = fusion_accuracy - model1_accuracy
    improvement_vs_model2 = fusion_accuracy - model2_accuracy

    print(f"\n置信度融合模型相比模型1的改进: {improvement_vs_model1:.4f}")
    print(f"置信度融合模型相比模型2的改进: {improvement_vs_model2:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(df['label'], df['融合预测标签'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('置信度融合模型混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix_fusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("混淆矩阵已保存为 'confusion_matrix_fusion.png'")

    # 绘制权重分布图
    plt.figure(figsize=(10, 6))
    plt.hist(df['模型1权重'], alpha=0.5, label='模型1权重')
    plt.hist(df['模型2权重'], alpha=0.5, label='模型2权重')
    plt.title('模型权重分布')
    plt.xlabel('权重')
    plt.ylabel('频次')
    plt.legend()
    plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("权重分布图已保存为 'weight_distribution.png'")

    # 添加随机森林融合
    rf_accuracy, rf_predictions, rf_proba = random_forest_fusion(df)

    # 将随机森林融合结果添加到DataFrame
    df['随机森林融合预测'] = rf_predictions
    df['随机森林融合置信度'] = np.max(rf_proba, axis=1)

    # 计算随机森林融合的改进
    improvement_rf_vs_model1 = rf_accuracy - model1_accuracy
    improvement_rf_vs_model2 = rf_accuracy - model2_accuracy
    improvement_rf_vs_fusion = rf_accuracy - fusion_accuracy

    print(f"\n随机森林融合模型相比模型1的改进: {improvement_rf_vs_model1:.4f}")
    print(f"随机森林融合模型相比模型2的改进: {improvement_rf_vs_model2:.4f}")
    print(f"随机森林融合模型相比置信度融合的改进: {improvement_rf_vs_fusion:.4f}")

    # 保存结果到Excel
    df.to_excel(output_file, index=False)
    print(f"\n融合结果已保存到: {output_file}")

    return df


# 使用示例
if __name__ == '__main__':
    input_file = r"D:\waibao\cancer_detect\predict_data\融合前.xlsx"  # 请替换为您的输入Excel文件路径
    output_file = r"D:\waibao\cancer_detect\predict_data\融合结果.xlsx"  # 请替换为您想要保存的输出文件路径

    # 执行融合
    result_df = apply_fusion_to_dataframe(input_file, output_file)

    # 显示融合后的前几行结果
    print("\n融合后的数据前几行:")
    print(result_df[['label', 'model1_result', 'model1_confidence',
                     'model2_result', 'model2_confidence',
                     '融合预测标签', '融合置信度', '模型1权重', '模型2权重',
                     '随机森林融合预测', '随机森林融合置信度']].head(10))