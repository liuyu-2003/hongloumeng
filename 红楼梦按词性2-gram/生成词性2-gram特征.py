import os
import jieba
import jieba.posseg as pseg
import re

data_folder = 'data'
output_folder = 'output_pos'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = pseg.cut(text)
    pos_tags = [word.flag for word in words if re.match(r'\w', word.word)]  # 去掉标点符号
    pos_ngrams = [" ".join(pos_tags[i:i+2]) for i in range(len(pos_tags) - 1)]

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-pos-2grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for pos_ngram in pos_ngrams:
            output_file.write(pos_ngram + '\n')

print("词性2-gram 特征生成完成")
