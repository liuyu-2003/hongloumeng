import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties

output_folder = 'output_char_1gram'
tfidf_file_path = os.path.join(output_folder, 'char_1grams_tfidf.json')

# 加载 TF-IDF 数据
with open(tfidf_file_path, 'r', encoding='utf-8') as file:
    tfidf_data = json.load(file)

# 获取所有章节的名称
chapters = [f'chapter-{i}' for i in range(1, 121)]

# 获取所有的1-gram特征
all_ngrams = list(tfidf_data.keys())

# 构建TF-IDF矩阵
tfidf_matrix = np.zeros((120, len(all_ngrams)))
for i, chapter in enumerate(chapters):
    for j, ngram in enumerate(all_ngrams):
        if chapter in tfidf_data[ngram]:
            tfidf_matrix[i, j] = tfidf_data[ngram][chapter]

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(tfidf_matrix)

# 执行K-means聚类
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(reduced_data)

# 设置中文字体
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # 根据你的系统路径设置
font_prop = FontProperties(fname=font_path)

# 生成散点图并标注K-means聚类结果
plt.figure(figsize=(10, 8))
colors = ['red' if label == 0 else 'blue' for label in labels]

# 绘制每个点并根据K-means结果着色
for i in range(len(labels)):
    plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=colors[i])

plt.title('《红楼梦》章节的t-SNE降维和K-means聚类结果 (k=2)', fontproperties=font_prop)
plt.xlabel('t-SNE 1', fontproperties=font_prop)
plt.ylabel('t-SNE 2', fontproperties=font_prop)
plt.grid(True)

# 保存散点图
plt.savefig(os.path.join(output_folder, 'tsne_kmeans_char_1grams_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE 降维和 K-means 聚类完成")
