from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_marianmt_model():
    return AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

def get_marianmt_tokenizer():
    return AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

