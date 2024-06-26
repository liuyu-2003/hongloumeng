import os
import re

data_folder = 'data'
output_folder = 'output_char_1gram'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 去掉标点符号和空格
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\w]', '', text)

    n_grams = list(text)  # 1-gram就是每个单独的字符

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-1grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for n_gram in n_grams:
            output_file.write(n_gram + '\n')

print("1-gram 特征生成完成")
