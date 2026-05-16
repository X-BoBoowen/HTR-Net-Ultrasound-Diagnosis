# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import precision_recall_curve
# import shap
# import warnings
# import joblib  # 用于保存和加载模型
#
# warnings.filterwarnings('ignore')
#
# # ==================== 数据准备 ====================
# # 读取Excel文件
# df = pd.read_excel('D:\waibao\cancer_detect\data.xlsx')
#
# # 添加GLS过滤条件 - 只保留GLS > -15%的数据
# print(f"原始数据量: {len(df)}")
# df = df[df['LV-GLS'] > -15]  # 假设特征名为
# print(f"过滤后数据量(GLS > -15%): {len(df)}")
#
# # 特征工程
# features = df.columns[:-1]
# X = df[features].values
# y = df['label'].values
#
# # 缺失值填充 - 使用更高级的迭代插补
# imputer = IterativeImputer(random_state=42, max_iter=10)
# X = imputer.fit_transform(X)
#
# # 添加特征交互项（提高模型非线性能力）
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X)
# X = np.hstack([X, X_poly[:, X.shape[1]:]])  # 只保留交互项
#
# # 数据标准化 - 使用RobustScaler处理异常值
# scaler = RobustScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y,
#     test_size=0.1,
#     stratify=y,
#     random_state=42
# )
#
# # ==================== 模型训练优化 ====================
# # 计算类别权重
# neg = sum(y_train == 0)
# pos = sum(y_train == 1)
# scale_pos_weight = neg / pos
#
# # 使用贝叶斯优化进行超参数调优
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer
#
# # 定义搜索空间
# search_spaces = {
#     'learning_rate': Real(0.001, 0.2, 'log-uniform'),
#     'max_depth': Integer(3, 8),
#     'n_estimators': Integer(100, 500),
#     'gamma': Real(0, 0.5),
#     'min_child_weight': Integer(1, 10),
#     'subsample': Real(0.6, 1.0),
#     'colsample_bytree': Real(0.6, 1.0),
#     'reg_alpha': Real(0, 1),
#     'reg_lambda': Real(0, 1)
# }
#
# # 创建基础模型
# base_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     scale_pos_weight=scale_pos_weight,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='auc',
#     early_stopping_rounds=20
# )
#
# # 贝叶斯优化搜索
# bayes_search = BayesSearchCV(
#     base_model,
#     search_spaces,
#     n_iter=50,  # 增加迭代次数
#     cv=StratifiedKFold(5, shuffle=True, random_state=42),
#     scoring='roc_auc',
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )
#
# bayes_search.fit(X_train, y_train,
#                  eval_set=[(X_test, y_test)],  # 测试集仅用于早停
#                  verbose=False)
#
# best_model = bayes_search.best_estimator_
# print(f"Best parameters: {bayes_search.best_params_}")
# print(f"Best CV AUC: {bayes_search.best_score_:.4f}")
#
# # 使用模型堆叠进一步提升性能
# stacked_model = StackingClassifier(
#     estimators=[
#         ('xgb1', xgb.XGBClassifier(**bayes_search.best_params_,
#                                    objective='binary:logistic',
#                                    random_state=42,
#                                    use_label_encoder=False)),
#         ('xgb2', xgb.XGBClassifier(learning_rate=0.05,
#                                    max_depth=4,
#                                    n_estimators=300,
#                                    random_state=42,
#                                    use_label_encoder=False))
#     ],
#     final_estimator=LogisticRegression(penalty='l2', C=0.1, solver='liblinear'),
#     cv=5,
#     n_jobs=-1
# )
#
# stacked_model.fit(X_train, y_train)
#
# # ==================== 使用训练集进行评估 ====================
# # 评估两个模型（使用训练集）
# y_pred_proba_base_train = best_model.predict_proba(X_train)[:, 1]
# auc_base_train = roc_auc_score(y_train, y_pred_proba_base_train)
#
# y_pred_proba_stacked_train = stacked_model.predict_proba(X_train)[:, 1]
# auc_stacked_train = roc_auc_score(y_train, y_pred_proba_stacked_train)
#
# # 选择表现更好的模型
# if auc_stacked_train > auc_base_train:
#     final_model = stacked_model
#     y_score_train = y_pred_proba_stacked_train
#     print(f"Using Stacked Model ( AUC = {auc_stacked_train:.4f})")
# else:
#     final_model = best_model
#     y_score_train = y_pred_proba_base_train
#     print(f"Using Base Model ( AUC = {auc_base_train:.4f})")
#
# # 寻找最佳概率阈值（使用训练集）
# fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
# optimal_idx_train = np.argmax(tpr_train - fpr_train)
# optimal_threshold_train = thresholds_train[optimal_idx_train]
# print(f"Optimal probability threshold (): {optimal_threshold_train:.3f}")
#
# y_pred_train = (y_score_train >= optimal_threshold_train).astype(int)
# roc_auc_train = auc(fpr_train, tpr_train)
#
# # ==================== 保存模型和预处理对象 ====================
# # 保存最终模型
# joblib.dump(final_model, 'final_model.pkl')
# # 保存预处理对象
# joblib.dump(imputer, 'imputer.pkl')
# joblib.dump(poly, 'poly.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("模型和预处理对象已保存")
#
#
# # ==================== 使用模型预测新数据 ====================
# def predict_new_data(excel_file_path, model, imputer, poly, scaler, features):
#     """
#     使用训练好的模型预测新数据
#
#     参数:
#     excel_file_path: Excel文件路径
#     model: 训练好的模型
#     imputer: 缺失值填充器
#     poly: 多项式特征生成器
#     scaler: 标准化器
#     features: 特征列名列表
#
#     返回:
#     包含预测结果的DataFrame
#     """
#     # 读取新数据
#     new_df = pd.read_excel(excel_file_path)
#
#     # 检查是否包含必要的特征
#     missing_features = set(features) - set(new_df.columns)
#     if missing_features:
#         print(f"警告: 数据中缺少以下特征: {missing_features}")
#         return None
#
#     # 提取特征数据
#     X_new = new_df[features].values
#
#     # 应用相同的预处理步骤
#     X_new_imputed = imputer.transform(X_new)
#     X_new_poly = poly.transform(X_new_imputed)
#     X_new_final = np.hstack([X_new_imputed, X_new_poly[:, X_new_imputed.shape[1]:]])
#     X_new_scaled = scaler.transform(X_new_final)
#
#     # 进行预测
#     predictions = model.predict(X_new_scaled)
#     probabilities = model.predict_proba(X_new_scaled)[:, 1]
#
#     # 将预测结果添加到DataFrame
#     result_df = new_df.copy()
#     result_df['预测标签'] = predictions
#     result_df['预测概率'] = probabilities
#
#     return result_df
#
#
# # 使用模型预测新数据
# new_data_path = 'D:\waibao\cancer_detect\data.xlsx'  # 替换为你的新数据路径
# try:
#     prediction_results = predict_new_data(
#         new_data_path,
#         final_model,
#         imputer,
#         poly,
#         scaler,
#         features
#     )
#
#     if prediction_results is not None:
#         # 保存预测结果
#         output_path = 'D:\waibao\cancer_detect\prediction_results.xlsx'
#         prediction_results.to_excel(output_path, index=False)
#         print(f"预测结果已保存到: {output_path}")
#
#         # 显示预测结果摘要
#         print("\n预测结果摘要:")
#         print(f"总样本数: {len(prediction_results)}")
#         print(f"预测为阳性(HER2+)的样本数: {sum(prediction_results['预测标签'] == 1)}")
#         print(f"预测为阴性(HER2-)的样本数: {sum(prediction_results['预测标签'] == 0)}")
#
#         # 显示前几个预测结果
#         print("\n前10个样本的预测结果:")
#         print(prediction_results[['预测标签', '预测概率']].head(10))
#
# except Exception as e:
#     print(f"预测过程中发生错误: {str(e)}")
#
# # ==================== 可视化部分（使用训练集数据） ====================
# # ... (保持原有的可视化代码不变)
#
# # ==================== 输出报告 ====================
# print("\nClassification Report ( Set):")
# print(classification_report(y_train, y_pred_train, target_names=['HER2-', 'HER2+']))
#
# print(f"\nOptimized Model  AUC: {roc_auc_train:.4f}")
# print(f"Best Model Parameters: {bayes_search.best_params_}")
#
# # 特征重要性表格（使用SHAP值）
# # 注意：这里我们仍然使用之前的SHAP值计算方法
# explainer_old = shap.TreeExplainer(best_model)
# shap_values_old = explainer_old.shap_values(X_train)
# shap_values_mean = np.abs(shap_values_old).mean(0)
# sorted_idx = np.argsort(shap_values_mean)[::-1][:7]  # 只取前7个最重要的特征
#
# importance_df = pd.DataFrame({
#     'Feature': [f"Feature_{i}" for i in sorted_idx],
#     'Importance': shap_values_mean[sorted_idx]
# }).sort_values('Importance', ascending=False)
#
# print("\nTop Feature Importance Table:")
# print(importance_df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
import shap
import warnings
import joblib  # 用于保存和加载模型

warnings.filterwarnings('ignore')

# ==================== 数据准备 ====================
# 读取Excel文件
df = pd.read_excel('D:\waibao\cancer_detect\data.xlsx')

# 添加GLS过滤条件 - 只保留GLS > -15%的数据
print(f"原始数据量: {len(df)}")
df = df[df['LV-GLS'] > -15]  # 假设特征名为
print(f"过滤后数据量(GLS > -15%): {len(df)}")

# 特征工程
features = df.columns[:-1]
X = df[features].values
y = df['label'].values

# 缺失值填充 - 使用更高级的迭代插补
imputer = IterativeImputer(random_state=42, max_iter=10)
X = imputer.fit_transform(X)

# 添加特征交互项（提高模型非线性能力）
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X = np.hstack([X, X_poly[:, X.shape[1]:]])  # 只保留交互项

# 数据标准化 - 使用RobustScaler处理异常值
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

# ==================== 模型训练优化 ====================
# 计算类别权重
neg = sum(y_train == 0)
pos = sum(y_train == 1)
scale_pos_weight = neg / pos

# 使用贝叶斯优化进行超参数调优
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 定义搜索空间
search_spaces = {
    'learning_rate': Real(0.001, 0.2, 'log-uniform'),
    'max_depth': Integer(3, 8),
    'n_estimators': Integer(100, 500),
    'gamma': Real(0, 0.5),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0, 1)
}

# 创建基础模型
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=20
)

# 贝叶斯优化搜索
bayes_search = BayesSearchCV(
    base_model,
    search_spaces,
    n_iter=50,  # 增加迭代次数
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

bayes_search.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],  # 测试集仅用于早停
                 verbose=False)

best_model = bayes_search.best_estimator_
print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best CV AUC: {bayes_search.best_score_:.4f}")

# 使用模型堆叠进一步提升性能
stacked_model = StackingClassifier(
    estimators=[
        ('xgb1', xgb.XGBClassifier(**bayes_search.best_params_,
                                   objective='binary:logistic',
                                   random_state=42,
                                   use_label_encoder=False)),
        ('xgb2', xgb.XGBClassifier(learning_rate=0.05,
                                   max_depth=4,
                                   n_estimators=300,
                                   random_state=42,
                                   use_label_encoder=False))
    ],
    final_estimator=LogisticRegression(penalty='l2', C=0.1, solver='liblinear'),
    cv=5,
    n_jobs=-1
)

stacked_model.fit(X_train, y_train)

# ==================== 使用训练集进行评估 ====================
# 评估两个模型（使用训练集）
y_pred_proba_base_train = best_model.predict_proba(X_train)[:, 1]
auc_base_train = roc_auc_score(y_train, y_pred_proba_base_train)

y_pred_proba_stacked_train = stacked_model.predict_proba(X_train)[:, 1]
auc_stacked_train = roc_auc_score(y_train, y_pred_proba_stacked_train)

# 选择表现更好的模型
if auc_stacked_train > auc_base_train:
    final_model = stacked_model
    y_score_train = y_pred_proba_stacked_train
    print(f"Using Stacked Model ( AUC = {auc_stacked_train:.4f})")
else:
    final_model = best_model
    y_score_train = y_pred_proba_base_train
    print(f"Using Base Model ( AUC = {auc_base_train:.4f})")

# 寻找最佳概率阈值（使用训练集）
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
optimal_idx_train = np.argmax(tpr_train - fpr_train)
optimal_threshold_train = thresholds_train[optimal_idx_train]
print(f"Optimal probability threshold (): {optimal_threshold_train:.3f}")

y_pred_train = (y_score_train >= optimal_threshold_train).astype(int)
roc_auc_train = auc(fpr_train, tpr_train)

# ==================== 保存模型和预处理对象 ====================
# 保存最终模型
joblib.dump(final_model, 'final_model.pkl')
# 保存预处理对象
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(poly, 'poly.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("模型和预处理对象已保存")


# ==================== 使用模型预测新数据 ====================
def predict_new_data(excel_file_path, model, imputer, poly, scaler, features):
    """
    使用训练好的模型预测新数据

    参数:
    excel_file_path: Excel文件路径
    model: 训练好的模型
    imputer: 缺失值填充器
    poly: 多项式特征生成器
    scaler: 标准化器
    features: 特征列名列表

    返回:
    包含预测结果的DataFrame
    """
    # 读取新数据
    new_df = pd.read_excel(excel_file_path)

    # 检查是否包含必要的特征
    missing_features = set(features) - set(new_df.columns)
    if missing_features:
        print(f"警告: 数据中缺少以下特征: {missing_features}")
        return None

    # 提取特征数据
    X_new = new_df[features].values

    # 应用相同的预处理步骤
    X_new_imputed = imputer.transform(X_new)
    X_new_poly = poly.transform(X_new_imputed)
    X_new_final = np.hstack([X_new_imputed, X_new_poly[:, X_new_imputed.shape[1]:]])
    X_new_scaled = scaler.transform(X_new_final)

    # 进行预测
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    # 将预测结果添加到DataFrame
    result_df = new_df.copy()
    result_df['预测标签'] = predictions
    result_df['预测概率'] = probabilities

    return result_df


# 使用模型预测新数据
new_data_path = 'D:\waibao\cancer_detect\data.xlsx'  # 替换为你的新数据路径
try:
    prediction_results = predict_new_data(
        new_data_path,
        final_model,
        imputer,
        poly,
        scaler,
        features
    )

    if prediction_results is not None:
        # 保存预测结果
        output_path = 'D:\waibao\cancer_detect\prediction_results.xlsx'
        prediction_results.to_excel(output_path, index=False)
        print(f"预测结果已保存到: {output_path}")

        # 显示预测结果摘要
        print("\n预测结果摘要:")
        print(f"总样本数: {len(prediction_results)}")
        print(f"预测为阳性(HER2+)的样本数: {sum(prediction_results['预测标签'] == 1)}")
        print(f"预测为阴性(HER2-)的样本数: {sum(prediction_results['预测标签'] == 0)}")

        # 显示前几个预测结果
        print("\n前10个样本的预测结果:")
        print(prediction_results[['预测标签', '预测概率']].head(10))

except Exception as e:
    print(f"预测过程中发生错误: {str(e)}")

# ==================== 可视化部分（使用训练集数据） ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# 1. 特征相关性热力图（原始特征）
plt.figure(figsize=(10, 8))
corr_matrix = df[features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={'label': 'Correlation Coefficient'}, vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. 特征分布小提琴图
plt.figure(figsize=(14, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)  # 根据特征数量调整行列数（此处为2x4布局）

    # 绘制小提琴图
    sns.violinplot(
        x='label', y=feature,
        data=df,
        palette={0: 'royalblue', 1: 'crimson'},
        split=True,
        inner="quartile",
        cut=0
    )

    # 美化格式
    plt.title(f'{feature}', fontsize=12)
    plt.xlabel('HER2 Status' if i > 4 else '', fontsize=10)
    plt.ylabel('')
    plt.yticks([])

    # 调整x轴刻度样式
    plt.xticks(
        ticks=[0, 1],
        labels=['Negative', 'Positive'],
        rotation=45,
        ha='right',
        fontsize=9
    )
    plt.grid(axis='y', alpha=0.3)

# 修改主标题样式
plt.suptitle("FEATURE DISTRIBUTION COMPARISON BY HER2 STATUS",
             y=1.02,
             fontsize=16,
             weight='bold',
             fontname='Times New Roman')
plt.tight_layout()
plt.show()

# 3. 决策边界可视化（使用原始特征）
# 注意：这里使用原始特征进行PCA，而不是包含交互项的特征
pca = PCA(n_components=2)
X_pca_original = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(df[features].values)))

# 训练简化模型
model_pca = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.1,
    n_estimators=50,
    max_depth=2,
    random_state=42,
    use_label_encoder=False
)
model_pca.fit(X_pca_original, y)

# 生成网格数据
x_min, x_max = X_pca_original[:, 0].min() - 1, X_pca_original[:, 0].max() + 1
y_min, y_max = X_pca_original[:, 1].min() - 1, X_pca_original[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测概率
Z = model_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, levels=20, cmap='RdYlBu')
scatter = plt.scatter(X_pca_original[:, 0], X_pca_original[:, 1], c=y,
                      edgecolors='k', cmap='RdYlBu', s=50)
plt.colorbar(scatter, label='Probability')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title("XGBoost Decision Boundary (PCA Projection)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 4. 混淆矩阵（优化后模型在训练集上的表现）
plt.figure(figsize=(8, 6))
cm_train = confusion_matrix(y_train, y_pred_train)
sns.heatmap(
    cm_train,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.title('Optimized Model Confusion Matrix ( Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 5. ROC曲线（优化后模型在训练集上的表现）
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc_train:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Optimized XGBoost ROC Curve ( Set)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 7. 概率分布直方图（优化后模型在训练集上的表现）
plt.figure(figsize=(10, 6))
plt.hist(y_score_train[y_train == 0], bins=30, alpha=0.6,
         color='royalblue', edgecolor='k', density=True, label='Class 0')
plt.hist(y_score_train[y_train == 1], bins=30, alpha=0.6,
         color='crimson', edgecolor='k', density=True, label='Class 1')
plt.axvline(x=optimal_threshold_train, color='green', linestyle='--',
            label=f'Optimal Threshold: {optimal_threshold_train:.2f}')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title(f'Probability Distribution ( AUC = {roc_auc_train:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 8. 模型校准曲线（使用训练集）
plt.figure(figsize=(8, 6))
prob_true_train, prob_pred_train = calibration_curve(y_train, y_score_train, n_bins=10)
plt.plot(prob_pred_train, prob_true_train, marker='o', label='Optimized Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve ( Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 9. 精度-召回曲线（使用训练集）
plt.figure(figsize=(8, 6))
precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
plt.plot(recall_train, precision_train, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve ( Set)')
plt.grid(alpha=0.3)
plt.show()

# ==================== 输出报告 ====================
print("\nClassification Report ( Set):")
print(classification_report(y_train, y_pred_train, target_names=['HER2-', 'HER2+']))

print(f"\nOptimized Model  AUC: {roc_auc_train:.4f}")
print(f"Best Model Parameters: {bayes_search.best_params_}")

# 特征重要性表格（使用SHAP值）
# 注意：这里我们仍然使用之前的SHAP值计算方法
explainer_old = shap.TreeExplainer(best_model)
shap_values_old = explainer_old.shap_values(X_train)
shap_values_mean = np.abs(shap_values_old).mean(0)
sorted_idx = np.argsort(shap_values_mean)[::-1][:7]  # 只取前7个最重要的特征

importance_df = pd.DataFrame({
    'Feature': [f"Feature_{i}" for i in sorted_idx],
    'Importance': shap_values_mean[sorted_idx]
}).sort_values('Importance', ascending=False)

print("\nTop Feature Importance Table:")
print(importance_df)