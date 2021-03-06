from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from utils.dataset import YandexRuEnDataset, collate_sentences
from models.marianmt import get_marianmt_model, get_marianmt_tokenizer

if __name__ == '__main__':
    tokenizer = get_marianmt_tokenizer()
    model = get_marianmt_model()
    train_dataset = YandexRuEnDataset("data", split="train")
    val_dataset = YandexRuEnDataset("data", split="valid")

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs

    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=collate_sentences(tokenizer)
    )

    trainer.train()