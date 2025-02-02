import os
import csv

from glob import glob


file_format = '.txt'
search_path = 'fl/GPT_OUTPUT/'
out_file = 'fl/GPT_OUTPUT/gpt_answers_processed.csv'

rows = [['iter', 'ID', 'pos', 'text', 'FT', 'confidence']]

file_paths = glob(os.path.join(search_path, '*'+file_format), recursive=False)
print(file_paths)

for file_path in file_paths:

    issue_id = file_path.split(os.path.sep)[-1].replace(file_format, '')
    print(issue_id)
    iter = -1

    with open(file_path) as f:
        lines = f.readlines()

        for x in lines:
            if x[0].isdigit():
                dot_pos = x.find('.')

                if dot_pos == -1:
                    continue

                answer_pos = int(x[:dot_pos])
                text = x[dot_pos+1:]

                if answer_pos == 1:
                    iter+=1

                rows.append([iter, issue_id, answer_pos, text, '', ''])



with open(out_file, 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n', )
    writer.writerows(rows)
