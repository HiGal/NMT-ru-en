import torch
from tqdm import tqdm

if __name__ == '__main__':

    ru2en = torch.hub.load("pytorch/fairseq", "transformer.wmt19.ru-en",
                           checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                           tokenizer='moses', bpe='fastbpe')
    ru2en.cuda()
    translations = []
    with open("data/test.ru.txt", "r") as f:
        src = f.readlines()
    for line in tqdm(src):
        res = ru2en.translate(line)
        translations.append(res+"\n")

    with open("answer.txt","w") as f:
        f.writelines(translations)
