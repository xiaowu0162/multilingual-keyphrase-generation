import json
from tqdm import tqdm


'''
for split in ['train', 'dev', 'test']:
    in_file = f'original/{split}.json'
    out_file = f'mix.{split}.json'

    with open(in_file) as in_f, open(out_file, 'w') as out_f:
        for line in tqdm(in_f.readlines(), desc=split):
            entry = json.loads(line)
            entry['asin'] = entry['id']
            entry['content'] = '<title>' + entry['title'] + '<context>' + entry['context']
            out_f.write(json.dumps(entry) + '\n')
'''


#for split in ['train', 'dev', 'test']:
for split in ['dev', 'test']:
    in_file = f'en.{split}.json'
    out_file = f'en_only/mix.{split}.json'

    with open(in_file) as in_f, open(out_file, 'w') as out_f:
        for line in tqdm(in_f.readlines(), desc=split):
            entry = json.loads(line)
            entry['asin'] = entry['id']
            entry['content'] = '<title>' + entry['title'] + '<context>' + entry['context']
            out_f.write(json.dumps(entry) + '\n')
