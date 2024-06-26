import os
import re

def _char_ngrams(text, n=3):
    """生成字符的n-gram"""
    return [text[i:i+n] for i in range(len(text)-n+1)]

data_folder = 'data'
output_folder = 'output_char_3gram'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 去掉标点符号和空格
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\w]', '', text)

    n_grams = _char_ngrams(text, n=3)

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-3grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for n_gram in n_grams:
            output_file.write(n_gram + '\n')

print("3-gram 特征生成完成")
