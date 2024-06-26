import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

output_folder = 'output'
tfidf_file_path = os.path.join(output_folder, '3grams_tfidf.json')

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

# 执行PCA降维，增加主成分数量
pca = PCA(n_components=10)
reduced_data = pca.fit_transform(tfidf_matrix)

# 打印方差占比
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print(f"各主成分的方差占比: {explained_variance_ratio}")
print(f"累积方差占比: {cumulative_variance_ratio}")

# 设置中文字体
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # 根据你的系统路径设置
font_prop = FontProperties(fname=font_path)

# 生成散点图，仅使用前两个主成分
plt.figure(figsize=(10, 8))

# 前80回目用红色点表示
plt.scatter(reduced_data[:80, 0], reduced_data[:80, 1], color='red', label='前80回')

# 后40回目用蓝色点表示
plt.scatter(reduced_data[80:, 0], reduced_data[80:, 1], color='blue', label='后40回')

plt.title('《红楼梦》章节的PCA降维结果', fontproperties=font_prop)
plt.xlabel('PCA 1', fontproperties=font_prop)
plt.ylabel('PCA 2', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.grid(True)

# 保存散点图
plt.savefig(os.path.join(output_folder, 'pca_3grams_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

print("PCA 降维和散点图生成完成")
