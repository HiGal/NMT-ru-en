from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_fsmt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")
    return tokenizer


def get_fsmt_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-ru-en")
    return model
