import os
import re

data_folder = 'data'
output_folder = 'output_char_2gram'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def _char_ngrams(text, ngram_range=(1, 1)):
    min_n, max_n = ngram_range
    ngrams = []
    n = len(text)
    for i in range(n):
        for j in range(i + min_n, min(i + max_n + 1, n + 1)):
            ngrams.append(text[i:j])
    return ngrams

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')  # 确保文件扩展名正确
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chars = list(text)
    chars = [char for char in chars if re.match(r'\w', char)]  # 去掉标点符号
    n_gramChars = _char_ngrams(chars, ngram_range=(2, 2))

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-char-2grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for n_gram in n_gramChars:
            output_file.write(''.join(n_gram) + '\n')  # 将列表转换为字符串

print("2-gram 特征生成完成")
