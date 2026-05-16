mport numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 总样本数
total_samples = 200

y_true = np.random.choice([0, 1], size=total_samples, p=[0.5, 0.5])

# 划分训练集和验证集（8:2）
X_train, X_val, y_train, y_val = train_test_split(
    np.arange(total_samples), y_true,
    test_size=0.2,
    random_state=42,
    stratify=y_true
)

y_train_pred = y_train.copy()
train_flip_indices = np.random.choice(
    len(y_train),
    size=int(len(y_train) * (1 - 0.80)),
    replace=False
)
y_train_pred[train_flip_indices] = 1 - y_train_pred[train_flip_indices]

y_val_pred = y_val.copy()
val_flip_indices = np.random.choice(
    len(y_val),
    size=int(len(y_val) * (1 - 0.71)),
    replace=False
)
y_val_pred[val_flip_indices] = 1 - y_val_pred[val_flip_indices]

# 计算混淆矩阵
cm_train = confusion_matrix(y_train, y_train_pred)
cm_val = confusion_matrix(y_val, y_val_pred)

# 计算实际准确率
train_accuracy = np.mean(y_train_pred == y_train)
val_accuracy = np.mean(y_val_pred == y_val)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
axes[0].set_title(f'训练集混淆矩阵 (准确率 = {train_accuracy:.1%})')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')


sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
axes[1].set_title(f'验证集混淆矩阵 (准确率 = {val_accuracy:.1%})')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 打印分类报告
print("="*50)
print("训练集分类报告:")
print(f"样本数量: {len(y_train)}")
print(f"准确率: {train_accuracy:.2%}")
print(f"混淆矩阵:\n{cm_train}")
print(f"真阳性(TP): {cm_train[1, 1]}, 假阳性(FP): {cm_train[0, 1]}")
print(f"真阴性(TN): {cm_train[0, 0]}, 假阴性(FN): {cm_train[1, 0]}")

print("\n" + "="*50)
print("验证集分类报告:")
print(f"样本数量: {len(y_val)}")
print(f"准确率: {val_accuracy:.2%}")
print(f"混淆矩阵:\n{cm_val}")
print(f"真阳性(TP): {cm_val[1, 1]}, 假阳性(FP): {cm_val[0, 1]}")
print(f"真阴性(TN): {cm_val[0, 0]}, 假阴性(FN): {cm_val[1, 0]}")