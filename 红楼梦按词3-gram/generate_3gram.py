import jieba
import re

jieba.load_userdict("manual-dict.txt") # file_name 为文件类对象或自定义词典的路径

def _word_ngrams(tokens, stop_words=None, ngram_range=(1, 1)):
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]))

    return tokens

# 读取文件内容
with open('data/dream_of_red_chamber.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用jieba进行分词
words = list(jieba.cut(text))

# 去掉标点符号
words = [word for word in words if re.match(r'\w', word)]

# 生成2-gram特征
n_gramWords = _word_ngrams(tokens=words, ngram_range=(3, 3))

# 将2-gram特征写入文件
with open('output/3grams_output.txt', 'w', encoding='utf-8') as output_file:
    for n_gramWord in n_gramWords:
        output_file.write(n_gramWord + '\n')
