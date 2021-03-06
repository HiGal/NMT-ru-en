from tokenizers import Tokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer, MBartTokenizer, FSMTTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import load_dataset



class YandexRuEnDataset(Dataset):

    def __init__(self, root_data, split):
        src = open(f"{root_data}/corpus.en_ru.1m.ru", "r").readlines()
        tgt = open(f"{root_data}/corpus.en_ru.1m.en", "r").readlines()
        # src = src[:int(0.1*len(src))]
        # tgt = tgt[:int(0.1 * len(tgt))]
        X_train, X_test, y_train, y_test = train_test_split(src, tgt, test_size=0.33, random_state=228)
        if split == "train":
            self.src = X_train
            self.trg = y_train
        elif split == "valid":
            self.src = X_test
            self.trg = y_test

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        trg = self.trg[idx]

        return src, trg


def collate_sentences(tokenizer: Tokenizer):
    def collate_fn(batch):
        batch = list(zip(*batch))
        X_batch = list(batch[0])
        y_batch = list(batch[1])
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=X_batch, tgt_texts=y_batch,
            padding=True, truncation=True,
            return_tensors='pt'
        )
        return batch

    return collate_fn

def get_wmt19_dataset(split):
    dataset = load_dataset("wmt19", 'ru-en')

if __name__ == '__main__':
    dataset = load_dataset("wmt19", "ru-en", split='train[:20%]')
    print(dataset)