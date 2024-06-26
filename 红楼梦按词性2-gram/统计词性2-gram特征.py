import os
from collections import defaultdict
import json

output_folder = 'output_pos'
ngram_count = defaultdict(lambda: defaultdict(int))

for chapter in range(1, 121):
    ngram_file_path = os.path.join(output_folder, f'chapter-{chapter}-pos-2grams.txt')
    with open(ngram_file_path, 'r', encoding='utf-8') as ngram_file:
        for line in ngram_file:
            pos_ngram = line.strip()
            ngram_count[pos_ngram][f'chapter-{chapter}'] += 1

output_count_file_path = os.path.join(output_folder, 'pos_2grams_count.json')

with open(output_count_file_path, 'w', encoding='utf-8') as count_file:
    json.dump(ngram_count, count_file, ensure_ascii=False, indent=4)

print("词性2-gram 特征统计完成")
