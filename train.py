from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from models.mbart import Trainer
from utils.datasets import YandexRuEnDataset, collate_sentences

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    model = AutoModel.from_pretrained("facebook/mbart-large-cc25")
    train_dataset = YandexRuEnDataset("../data", split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_sentences(tokenizer))
    trainer = Trainer(model)
    trainer.train_one_epoch(train_dataloader)