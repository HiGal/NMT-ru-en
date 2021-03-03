import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, MBartForConditionalGeneration

from utils.datasets import YandexRuEnDataset, collate_sentences


class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # self.scheduler = scheduler
        # self.cfg = config
        #
        # self.n_epochs = config.trainig.n_epochs
        self.device = "cuda:0"
        self.model.to(self.device)
        self.epoch = 0

    def train_one_epoch(self, train_dataloader):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        i = 0
        for step, batch in pbar:
            self.model.zero_grad()
            input_ids = batch.input_ids.to(self.device)
            labels = batch.labels.to(self.device)
            out = self.model(input_ids=input_ids, labels=labels)
            loss = out.loss

            loss.backward()
            self.optimizer.step()

            print(loss.item())
            if step == 2:
                break

    def train(self):
        pass

    def validation(self):
        pass

    def save_checkpoint(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    tokenizer.model_max_length = 1
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    train_dataset = YandexRuEnDataset("../data", split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_sentences(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    trainer = Trainer(model, optimizer)
    trainer.train_one_epoch(train_dataloader)
