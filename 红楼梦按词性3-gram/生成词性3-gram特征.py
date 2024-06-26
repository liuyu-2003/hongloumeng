import os
import jieba
import jieba.posseg as pseg
import re

jieba.load_userdict("manual-dict.txt")  # 加载自定义词典

def _pos_ngrams(pos_tags, ngram_range=(1, 1)):
    """将词性标注转换为n-gram序列"""
    min_n, max_n = ngram_range
    if max_n != 1:
        original_pos_tags = pos_tags
        pos_tags = []
        n_original_tags = len(original_pos_tags)
        for n in range(min_n, min(max_n + 1, n_original_tags + 1)):
            for i in range(n_original_tags - n + 1):
                pos_tags.append(" ".join(original_pos_tags[i: i + n]))
    return pos_tags

data_folder = 'data'
output_folder = 'output_pos_3gram'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = pseg.cut(text)
    pos_tags = [tag for _, tag in words if re.match(r'\w', tag)]  # 去掉标点符号，仅保留词性
    pos_ngrams = _pos_ngrams(pos_tags=pos_tags, ngram_range=(3, 3))

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-pos-3grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for n_gram in pos_ngrams:
            output_file.write(n_gram + '\n')

print("3-gram特征生成完成")
