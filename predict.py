from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm

if __name__ == '__main__':
    model = AutoModelWithLMHead.from_pretrained("facebook/wmt19-ru-en")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")
    translation = pipeline("translation_ru_to_en", model=model, tokenizer=tokenizer, device=0)
    batch_size = 16
    translations = []
    with open("data/test.ru.txt", "r") as f:
        src = f.readlines()
        n_batches = len(src) // batch_size
    for i in tqdm(range(n_batches)):
        if i != n_batches-1:
            batch = src[i * batch_size:i * batch_size + batch_size]
        else:
            batch = src[i * batch_size:]
        # batch = tokenizer.prepare_seq2seq_batch(batch)
        translated_text = translation(batch)
        tmp = [translated['translation_text']+"\n" for translated in translated_text]
        translations += tmp

    with open("answer.txt","w") as f:
        f.writelines(translations)
    # text = "8. С учетом изложенного выше Япония хотела бы предложить следующие элементы заключительных документов обзорной конференции 2015 года для дальнейшего обсуждения государствами-участниками."
    # translated_text = translation(text)[0]['translation_text']
    # print(translated_text)
