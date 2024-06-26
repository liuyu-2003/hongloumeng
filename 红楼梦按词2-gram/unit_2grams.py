import os
import jieba
import re

jieba.load_userdict("manual-dict.txt") # file_name 为文件类对象或自定义词典的路径

def _word_ngrams(tokens, ngram_range=(1, 1)):
    """Turn tokens into a sequence of n-grams"""
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]))
    return tokens

data_folder = 'data'
output_folder = 'output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for chapter in range(1, 121):
    file_path = os.path.join(data_folder, f'chapter-{chapter}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = list(jieba.cut(text))
    words = [word for word in words if re.match(r'\w', word)]  # 去掉标点符号
    n_gramWords = _word_ngrams(tokens=words, ngram_range=(2, 2))

    output_file_path = os.path.join(output_folder, f'chapter-{chapter}-2grams.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for n_gram in n_gramWords:
            output_file.write(n_gram + '\n')

print("2-gram 特征生成完成")
