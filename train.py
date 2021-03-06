from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from utils.dataset import YandexRuEnDataset, collate_sentences
from models.marianmt import get_marianmt_model, get_marianmt_tokenizer
from utils.model_utils import freeze_params, freeze_encoder, freeze_embeds
from models.fsmt import get_fsmt_model, get_fsmt_tokenizer

if __name__ == '__main__':
    tokenizer = get_marianmt_tokenizer()
    tokenizer.model_max_length=100
    model = get_marianmt_model()
    freeze_embeds(model)
    freeze_encoder(model)
    train_dataset = YandexRuEnDataset("data", split="train")
    val_dataset = YandexRuEnDataset("data", split="valid")

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=5000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        fp16=True,
        fp16_opt_level='O2',
        save_steps=20000
    )

    trainer = Seq2SeqTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=collate_sentences(tokenizer)
    )

    trainer.train()