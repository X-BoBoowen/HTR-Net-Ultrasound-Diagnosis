import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pointbiserialr, chi2_contingency
import seaborn as sns

# 数据预处理
def preprocess_data(file_path):
    # 读取数据
    df = pd.read_excel(file_path)

    # 检查数据类型
    print("原始数据类型：\n", df.dtypes)

    # 处理异常值和非数值数据
    numeric_cols = ['年龄', 'Nt-proBNP', 'IVS-thickness',
                    'LVEDD', 'LAVImax', 'LVEF', 'LV-GLS']

    for col in numeric_cols:
        # 移除特殊字符并转换为数值
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')

    # 处理缺失值
    df.dropna(inplace=True)

    # 验证标签分布
    print("\n标签分布：\n", df['Label'].value_counts())

    return df


# 特征重要性分析
def xgb_feature_analysis(df, target_col='Label', n_iter=10):
    # 分割特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 初始化存储结果
    importance_results = []
    metric_results = []

    for i in range(n_iter):
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=i
        )

        # 模型训练
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            random_state=i
        )

        model.fit(X_train, y_train)

        # 预测评估
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 存储指标
        metric_results.append({
            'Iteration': i + 1,
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba)
        })

        # 存储特征重要性
        importance = model.get_booster().get_score(importance_type='weight')
        for feat, score in importance.items():
            importance_results.append({
                'Feature': feat,
                'Importance': score,
                'Iteration': i + 1
            })

    # 结果整合
    importance_df = pd.DataFrame(importance_results)
    metric_df = pd.DataFrame(metric_results)

    return importance_df, metric_df


# 统计相关性分析
def statistical_analysis(df, target_col='Label'):
    results = []

    # 分类变量（性别）使用卡方检验
    contingency_table = pd.crosstab(df['性别'], df[target_col])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    cramers_v = np.sqrt(chi2 / (df.shape[0] * (min(contingency_table.shape) - 1)))
    results.append({
        'Feature': '性别',
        'Correlation': cramers_v,
        'p-value': p,
        'Method': "Cramér's V"
    })

    # 连续变量使用点二列相关系数
    numeric_features = ['年龄', 'Nt-proBNP', 'IVS-thickness',
                        'LVEDD', 'LAVImax', 'LVEF', 'LV-GLS']

    for feat in numeric_features:
        r, p = pointbiserialr(df[feat], df[target_col])
        results.append({
            'Feature': feat,
            'Correlation': r,
            'p-value': p,
            'Method': "Point-Biserial"
        })

    return pd.DataFrame(results)


# 主程序
if __name__ == "__main__":
    # 数据预处理
    file_path =r'D:\waibao\unetdata_paper\data.xlsx'  # 修改为实际路径
    df = preprocess_data(file_path)

    # XGBoost特征重要性分析
    importance_df, metric_df = xgb_feature_analysis(df, n_iter=10)

    # 统计相关性分析
    stats_df = statistical_analysis(df)

    # 结果可视化
    plt.figure(figsize=(12, 6))

    # XGBoost特征重要性
    plt.subplot(1, 2, 1)
    avg_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values()
    avg_importance.plot(kind='barh', color='skyblue')
    plt.title('XGBoost Feature Importance (Average over 10 runs)')
    plt.xlabel('Importance Score (Weight)')

    # 统计相关性
    plt.subplot(1, 2, 2)
    stats_df.set_index('Feature')['Correlation'].sort_values().plot(
        kind='barh', color='salmon')
    plt.title('Statistical Correlation Analysis')
    plt.xlabel('Correlation Coefficient')

    plt.tight_layout()
    plt.savefig('combined_analysis.png', dpi=300)
    plt.show()

    # 生成报告
    final_report = stats_df.merge(
        avg_importance.reset_index().rename(columns={'Importance': 'XGB Importance'}),
        on='Feature'
    )

    # 添加显著性标记
    final_report['Significance'] = np.where(
        final_report['p-value'] < 0.05,
        'Significant (p < 0.05)',
        'Not Significant'
    )

    # 保存结果
    with pd.ExcelWriter('analysis_results.xlsx') as writer:
        final_report.to_excel(writer, sheet_name='综合报告', index=False)
        metric_df.to_excel(writer, sheet_name='模型性能', index=False)

    print("\n最终分析报告：")
    print(final_report[['Feature', 'Correlation', 'p-value', 'XGB Importance', 'Significance']])
    print("\n模型性能（10次运行平均）：")
    print(metric_df.mean())
    # 琴谱图可视化
    # 筛选连续变量
    continuous_features = ['年龄', 'Nt-proBNP', 'IVS-thickness',
                           'LVEDD', 'LAVImax', 'LVEF', 'LV-GLS']

    # 设置画布
    plt.figure(figsize=(15, 20))
    plt.suptitle("特征分布琴谱图分析", y=1.02, fontsize=14)

    # 创建子图画板
    n_cols = 3
    n_rows = int(np.ceil(len(continuous_features) / n_cols))
    plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # 遍历绘制每个特征
    for idx, feature in enumerate(continuous_features):
        plt.subplot(n_rows, n_cols, idx + 1)

        # 绘制琴谱图
        sns.violinplot(x='Label', y=feature, data=df,
                       palette="Set2", split=True, inner="quartile")

        # 添加统计标注
        median_val = df.groupby('Label')[feature].median()
        plt.text(0, median_val[0] + 0.05, f'Median: {median_val[0]:.2f}',
                 ha='center', color='blue')
        plt.text(1, median_val[1] + 0.05, f'Median: {median_val[1]:.2f}',
                 ha='center', color='orange')

        # 美化设置
        plt.title(f"{feature}分布", fontsize=12)
        plt.xlabel("Label")
        plt.ylabel(feature)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('feature_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

