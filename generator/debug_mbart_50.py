from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50TokenizerFast, AutoTokenizer

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO", use_fast=True)

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
    
print(labels)
print(tokenizer.decode(labels[0]))
print(model_inputs.input_ids[0])
print(tokenizer.decode(model_inputs.input_ids[0]))

out = model(**model_inputs, labels=labels) # forward pass

print(out.keys())


print('===============================================')


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="ro_RO", use_fast=True)

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
    
print(labels)
print(tokenizer.decode(labels[0]))

print(model_inputs.input_ids[0])
print(tokenizer.decode(model_inputs.input_ids[0]))

out = model(**model_inputs, labels=labels) # forward pass
print(out.keys())
