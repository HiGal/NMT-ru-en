import torch
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
from transformers.tokenization_utils_base import TruncationStrategy


def shorten_sentence(models, sentence, tokenizer, device):
    sentence_tensor = tokenizer(sentence,
                                add_special_tokens=True,
                                truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                                return_tensors='pt',
                                padding=False)
    input_ids = sentence_tensor['input_ids'].to(device)
    attention_mask = sentence_tensor['attention_mask'].to(device)
    out_scores = None
    for model in models:
        generations = model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1
        )
        if out_scores is None:
            out_scores = list(generations.scores)
        else:
            for i in range(len(generations.scores) - len(out_scores)):
                out_scores.append(torch.zeros_like(out_scores[0], device=device))
            for i, item in enumerate(generations.scores):
                out_scores[i] += item

    outputs = [o.argmax(1)[0] for o in out_scores]
    translated_sentence = tokenizer.decode(outputs)
    # remove start token
    return translated_sentence


if __name__ == '__main__':
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("results_256/checkpoint-6282")
    model_16 = AutoModelWithLMHead.from_pretrained("results_16/checkpoint-6282").to(device)
    model_32 = AutoModelWithLMHead.from_pretrained("results_32/checkpoint-6282").to(device)
    model_64 = AutoModelWithLMHead.from_pretrained("results_64/checkpoint-6282").to(device)
    model_128 = AutoModelWithLMHead.from_pretrained("results_128/checkpoint-6282").to(device)
    model_256 = AutoModelWithLMHead.from_pretrained("results_256/checkpoint-6282").to(device)
    models = [model_16, model_32, model_64, model_128, model_256]

    # translation = pipeline("translation_ru_to_en", model=model_16, tokenizer=tokenizer, device=0)
    translations = []
    with open("data/test.ru.txt", "r") as f:
        src = f.readlines()
    for line in tqdm(src):
        line = line.rstrip()
        translations.append(shorten_sentence(models, line, tokenizer, device=device))

    with open("answer.txt", "w") as f:
        f.writelines([text + '\n' for text in translations])
