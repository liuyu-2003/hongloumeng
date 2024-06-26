import os
import jieba
import jieba.posseg as pseg
import re

jieba.load_userdict("manual-dict.txt")  # 加载自定义词典

data_folder = 'data'
output_folder = 'output_pos_1gram'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = pseg.cut(text)
    pos_tags = [tag for _, tag in words if re.match(r'\w', tag)]  # 去掉标点符号，仅保留词性

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-pos-1grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for tag in pos_tags:
            output_file.write(tag + '\n')

print("仅包含词性的1-gram特征生成完成")
