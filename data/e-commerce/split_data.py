import os
import json
from tqdm import tqdm


input_dir = './mix_de_es_fr_it'


langs = ['de', 'es', 'fr', 'it']


for split in ['train', 'dev', 'test']:
    lang2data = {l: [] for l in langs}
    with open(f'{input_dir}/mix.{split}.json') as in_f:
        for line in tqdm(in_f.readlines(), desc=split):
            entry = json.loads(line)
            lang2data[entry['lang']].append(line)

    for l in langs:
        os.makedirs(f'{l}_only', exist_ok=True)
        with open(f'{l}_only/mix.{split}.json', 'w') as out_f:
            for d in tqdm(lang2data[l], desc=l):
                out_f.write(d)
                
    print({l: len(d) for l, d in lang2data.items()})

 
