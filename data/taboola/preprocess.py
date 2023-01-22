import os
import csv
import json


in_file = 'multilingual_landing_page_tags_19.csv'
out_file = 'multilingual_landing_page_tags_19.json'

os.makedirs('lang_splitted', exist_ok=True)

with open(in_file) as csvf:
        csvReader = csv.DictReader(csvf)
        data = []
        for row in csvReader:
                entry = {'id': row['video_id'],
                         'title': row['title'],
                         'abstract': row['description'],
                         'keyphrases': ' ; '.join([x.strip() for x in row['tags'].split(',')]),
                         'language': row['language']}
                data.append(entry)

data.sort(key=(lambda x: x['language']))
langs = sorted(set([x['language'] for x in data]))


# write all data into one json
with open(out_file, 'w') as out_f:
        for entry in data:
                print(json.dumps(entry, ensure_ascii=False), file=out_f)

# write lang-specific data
for lang in langs:
        nkp, npkp = [], []
        with open(f'lang_splitted/{lang}.json', 'w') as out_f:
                for entry in data:
                        if entry['language'] == lang:
                                print(json.dumps(entry, ensure_ascii=False), file=out_f)
                                kps = entry['keyphrases'].split(' ; ')
                                nkp.append(len(kps))
                                npkp.append(len([x for x in kps if x in entry['title'] + ' ' + entry['abstract']])/len(kps))
        print(f'{lang}: {sum(nkp)/len(nkp)} kp/doc, {sum(npkp)/len(npkp) * 1000000 // 1000 / 10}% pkp')

# write lang dict
with open('lang_splitted/lang_dicts.txt', 'w') as out_f:
        for lang in langs:
                print(lang, file=out_f)
