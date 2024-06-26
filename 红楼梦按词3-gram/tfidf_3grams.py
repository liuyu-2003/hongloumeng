import os
import json
import math
from collections import defaultdict

output_folder = 'output'
ngram_count_file_path = os.path.join(output_folder, '3grams_count.json')

with open(ngram_count_file_path, 'r', encoding='utf-8') as file:
    ngram_count = json.load(file)

# 计算每个 3-gram 的文档频率（DF）
df = defaultdict(int)
for n_gram, chapters in ngram_count.items():
    df[n_gram] = len(chapters)

# 总文档数（章节数）
N = 120

# 计算每个 3-gram 在每章中的 TF-IDF 值
tfidf = defaultdict(lambda: defaultdict(float))
for n_gram, chapters in ngram_count.items():
    for chapter, count in chapters.items():
        tf = count
        idf = math.log(N / (1 + df[n_gram]))
        tfidf[n_gram][chapter] = tf * idf

# 将结果写入文件
tfidf_output_file_path = os.path.join(output_folder, '3grams_tfidf.json')

with open(tfidf_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(tfidf, file, ensure_ascii=False, indent=4)

print("TF-IDF 计算完成")
