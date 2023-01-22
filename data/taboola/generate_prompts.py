import os
import json
from datetime import datetime


data_dir = './lang_splitted/'
prompt_dir = './prompts/'
os.makedirs(prompt_dir, exist_ok=True)
out_file = f'{prompt_dir}/prompts_{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'


LANG2NAME = {'af': 'Afrikaans', 'ca': 'Catalan', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'es': 'Spanish', 'fr': 'French', 'he': 'Hebrew', 'hi': 'Hindi', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'nl': 'Dutch', 'no': 'Norwegian', 'pt': 'Portuguese', 'ro': 'Romanian', 'sv': 'Swedish', 'th': 'Thai'}


# load data
data = {}
for lang in LANG2NAME:
    with open(f'{data_dir}/{lang}.json') as f:
        data[lang] = []
        for line in f.readlines():
            data[lang].append(json.loads(line))


# for sampling examples
def sample_examples(data_list, n):
    filtered_data_list = [x for x in data_list if x['abstract'].strip() != ""]
    return filtered_data_list[:n]


# generate prompts
prompts = {}
examples = {}
for lang, lang_name in LANG2NAME.items():
    cur_examples = sample_examples(data[lang], 5)
    cur_lang_prompt = f"Keyphrases are the phrases that summarize the most important information in a document. Given a document in {lang_name} and English, generate the keyphrases in {lang_name}.\n"
    for example in cur_examples:
        cur_lang_prompt += "\n"
        cur_lang_prompt += f"Document title: {example['title']}\n"
        cur_lang_prompt += f"Document body: {example['abstract']}\n"
        cur_lang_prompt += f"Keyphrases: {example['keyphrases']}\n"

    print(cur_lang_prompt)
    print("==========\n==========\n==========")
    prompts[lang] = cur_lang_prompt
    examples[lang] = [x['id'] for x in cur_examples]


# output
with open(out_file, 'w') as out_f:
    out_data = {'examples': examples, 'prompts': prompts}
    out_f.write(json.dumps(out_data, ensure_ascii=False, indent=4))
    print('output wrote to', out_file)
