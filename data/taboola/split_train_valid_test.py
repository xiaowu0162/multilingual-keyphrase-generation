import os
import json
from glob import glob

in_dir = 'lang_splitted/'
out_dir = 'train_test_splitted/'

n_test = 100
n_valid = 50
assert n_valid <= n_test // 2   # share valid and test data due to limited resource


def process_instance(line):
    entry = json.loads(line)

    out_entry = {}
    out_entry['id'] = entry['id']
    out_entry['asin'] = entry['id']
    out_entry['lang'] = entry['language']
    out_entry['title'] = entry['title']
    out_entry['content'] = entry['abstract']
    out_entry['keywords'] = entry['keyphrases']

    return out_entry


files = glob(in_dir + '/*.json')
print(files)
for all_data_file in files:
    lang_code = all_data_file.split('/')[-1][:2]
    print(lang_code)
    cur_out_dir = f'{out_dir}/{lang_code}/'
    os.makedirs(cur_out_dir, exist_ok=True)
    
    with open(all_data_file) as in_f:
        data = [line for line in in_f.readlines()]

    valid_data = data[:n_valid]
    test_data = data[:n_test]
    train_data = data[n_test:]

    with open(cur_out_dir +'/train.json', 'w') as out_f:
        for line in train_data:
            print(json.dumps(process_instance(line), ensure_ascii=False), file=out_f)
    with open(cur_out_dir +'/valid.json', 'w') as out_f:
        for line in valid_data:
            print(json.dumps(process_instance(line), ensure_ascii=False), file=out_f)
    with open(cur_out_dir +'/test.json', 'w') as out_f:
        for line in test_data:
            print(json.dumps(process_instance(line), ensure_ascii=False), file=out_f)  
