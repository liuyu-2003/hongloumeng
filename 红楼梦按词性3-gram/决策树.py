import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties

# 设置文件路径
output_folder = 'output_pos_3gram'
tfidf_file_path = os.path.join(output_folder, 'pos_3grams_tfidf.json')

# 加载 TF-IDF 数据
with open(tfidf_file_path, 'r', encoding='utf-8') as file:
    tfidf_data = json.load(file)

# 获取所有章节的名称
chapters = [f'chapter-{i}' for i in range(1, 121)]

# 获取所有的3-gram特征
all_ngrams = list(tfidf_data.keys())

# 构建TF-IDF矩阵
tfidf_matrix = np.zeros((120, len(all_ngrams)))
for i, chapter in enumerate(chapters):
    for j, ngram in enumerate(all_ngrams):
        if chapter in tfidf_data[ngram]:
            tfidf_matrix[i, j] = tfidf_data[ngram][chapter]

# 创建标签，前80回记为0，后40回记为1
labels = np.array([0]*80 + [1]*40)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.3, random_state=42)

# 训练决策树分类器
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 预测概率
y_prob = clf.predict_proba(X_test)[:, 1]

# 计算AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.4f}")

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 设置中文字体
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # 根据你的系统路径设置
font_prop = FontProperties(fname=font_path)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontproperties=font_prop)
plt.ylabel('True Positive Rate', fontproperties=font_prop)
plt.title('ROC Curve', fontproperties=font_prop)
plt.legend(loc="lower right", prop=font_prop)
plt.grid(True)

# 保存ROC曲线图
roc_curve_path = os.path.join(output_folder, 'roc_curve_pos_3gram.png')
plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
plt.show()

print("ROC曲线生成完成")

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(tfidf_matrix)

# 用决策树分类器对所有数据进行预测
labels = clf.predict(tfidf_matrix)

# 生成对比图
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# 左图：原始数据的t-SNE降维结果
ax[0].scatter(reduced_data[:80, 0], reduced_data[:80, 1], color='red', label='前80回')
ax[0].scatter(reduced_data[80:, 0], reduced_data[80:, 1], color='blue', label='后40回')
ax[0].set_title('原始数据的t-SNE降维结果', fontproperties=font_prop)
ax[0].set_xlabel('t-SNE 1', fontproperties=font_prop)
ax[0].set_ylabel('t-SNE 2', fontproperties=font_prop)
ax[0].legend(prop=font_prop)
ax[0].grid(True)

# 右图：决策树分类结果
colors = ['red' if label == 0 else 'blue' for label in labels]
for i in range(len(labels)):
    ax[1].scatter(reduced_data[i, 0], reduced_data[i, 1], color=colors[i])
ax[1].set_title('决策树分类结果', fontproperties=font_prop)
ax[1].set_xlabel('t-SNE 1', fontproperties=font_prop)
ax[1].set_ylabel('t-SNE 2', fontproperties=font_prop)
ax[1].grid(True)

# 保存对比图
comparison_plot_path = os.path.join(output_folder, 'tsne_decision_tree_pos_3grams_comparison.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print("对比图生成完成")
