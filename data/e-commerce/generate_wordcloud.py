import json
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from googletrans import Translator


langs = ['de', 'es', 'fr', 'it']
stopwords = {'de': stopwords.words('german') ,
             'es': stopwords.words('spanish'),
             'fr': stopwords.words('french'),
             'it': stopwords.words('italian')}


input_dir = './mix_de_es_fr_it'

translator = Translator()


def generate_wordcloud(word2freq, out_name=None):
    pass


#for split in ['train', 'dev', 'test']:
for split in ['train']:
    lang2data = {l: [[], [], []] for l in langs}   # title words, body wprds, keywords
    with open(f'{input_dir}/mix.{split}.json') as in_f:
        for line in tqdm(in_f.readlines(), desc=split):
            entry = json.loads(line)
            lang2data[entry['lang']][0] += [x for x in entry['title'].lower().split()
                                            if x not in stopwords[entry['lang']]]
            lang2data[entry['lang']][1] += [x for x in entry['context'].lower().split()
                                            if x not in stopwords[entry['lang']]]
            lang2data[entry['lang']][2] += [x for y in entry['keywords'].lower().split(';') for x in y.split() 
                                            if x not in stopwords[entry['lang']]]
    for lang in langs:
        # with open(f'{lang}.{split}.title.mostcommon100.csv', 'w') as f:
        #     f.write('Frequency,Word\n')
        #     #for word, freq in Counter(lang2data[lang][0]).most_common(100):
        #     for word in lang2data[lang][0]:
        #         f.write(f'1,{word}\n')
        # with open(f'{lang}.{split}.context.mostcommon100.csv', 'w') as f:
        #     f.write('Frequency,Word\n')
        #     #for word, freq in Counter(lang2data[lang][1]).most_common(100):
        #     #    f.write(f'{freq},{word}\n')
        #     for word in lang2data[lang][1]:
        #         f.write(f'1,{word}\n')
        #     #print(lang)
        #print(Counter(lang2data[lang][0]).most_common(100))
        
        print(lang)
        '''
        data = []
        for word, freq in tqdm(list(Counter(lang2data[lang][1]).most_common(100))):
            try:
                data.append((word, freq, translator.translate(word, src=lang, dest='en')))
            except:
                continue
        print(data)
        print(' ,'.join([x[-1] for x in data]))
        '''

        # freqwords = Counter(lang2data[lang][0] + lang2data[lang][1]).most_common(500)
        # with open(f'{lang}.{split}.freqwords.txt', 'w') as f:
        #     for w, freq in list(freqwords):
        #         f.write(w)
        #         f.write('\n')

        freqwords = Counter(lang2data[lang][2]).most_common(500)
        with open(f'{lang}.{split}.freqwords.kp.txt', 'w') as f:
            for w, freq in list(freqwords):
                f.write(w)
                f.write('\n')
        

    #wc = WordCloud(max_font_size=40).generate(' '.join(lang2data['de'][0]))
    #plt.figure()
    #plt.imsave(wc, 'test_wc.jpg')
    #plt.imshow(wordcloud, interpolation="bilinear")
    #plt.axis("off")
    #plt.show()
    

 
