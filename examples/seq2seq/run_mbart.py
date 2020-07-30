# from transformers import MBartTokenizer, BartForConditionalGeneration
#
# tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-en-ro', cache_dir='/Users/Mehrad/Documents/GitHub/genienlp/.embeddings')
# model = BartForConditionalGeneration.from_pretrained('facebook/mbart-large-en-ro', cache_dir='/Users/Mehrad/Documents/GitHub/genienlp/.embeddings')
#
# src_sent = "UN Chief Says There Is No Military Solution in Syria"
#
# src_ids = tokenizer.prepare_translation_batch([src_sent])
#
# output_ids = model.generate(src_ids["input_ids"], decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
#
# output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
#
# print('src_sent: ', src_sent)
# print('src_ids: ', src_ids)
# print('output_ids: ', output_ids)
# print('output: ', output)


from transformers import MBartTokenizer, BartForConditionalGeneration

tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25', cache_dir='/Users/Mehrad/Documents/GitHub/genienlp/.embeddings')
model = BartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25', cache_dir='/Users/Mehrad/Documents/GitHub/genienlp/.embeddings')

src_sent = "UN Chief Says There Is No Military Solution in Syria"

src_ids = tokenizer.prepare_translation_batch([src_sent])

output_ids = model.generate(src_ids["input_ids"])

output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print('src_sent: ', src_sent)
print('src_ids: ', src_ids)
print('output_ids: ', output_ids)
print('output: ', output)
