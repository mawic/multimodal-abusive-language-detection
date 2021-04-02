from torch.utils.data import DataLoader
import transformers as tff
from transformers import TrainingArguments, DistilBertForSequenceClassification, Trainer
from util.WeightedRandomSampler import WeightedRandomSampler
from dataset_parser.wich_parser import WichDataset
from dataset_parser.waseem_parser import WaseemDataset
from src.dataset_parser.davidson_parser import DavidsonDataset
import torch.nn as nn

class TweetBERT(nn.Module):
    def __init__(self, trained = False, freeze_weights = False, dataset="davidson"):
        super().__init__()
        self.dataset = dataset
        if trained:
            if dataset in "davidson":
                self.model : DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained('../../models/DistilBERT-davidson.model')
            elif dataset in "waseem":
                self.model: DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(
                    '../../models/DistilBERT-waseem.model')
            elif dataset in "wich":
                self.model: DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(
                    '../../models/DistilBERT-wich.model')
        else:
            self.MODEL_NAME = 'distilbert-base-uncased'
            if dataset in "davidson":
                self.model : DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=3) # set num_labels=3 so that for now the final layer predicts hate speech classes
            elif dataset in "waseem":
                self.model : DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=3) # set num_labels=3 so that for now the final layer predicts hate speech classes
            elif dataset in "wich":
                self.MODEL_NAME = "distilbert-base-german-cased"
                self.model : DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2) # set num_labels=3 so that for now the final layer predicts hate speech classes
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: tweet token ids
        :param attention_mask: tweet attention mask
        :return: result of the DistilBERT classifier layer without softmax.
        """
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask)[0] #for batchsize = 1

    def pretrain(self):
        if self.dataset in "waseem":
            train_dataset = WaseemDataset(model_name=self.MODEL_NAME, fixed_set="train")
            dev_dataset = WaseemDataset(model_name=self.MODEL_NAME, fixed_set="val")
        elif self.dataset in "davidson":
            train_dataset = DavidsonDataset(model_name=self.MODEL_NAME, fixed_set="train")
            dev_dataset = DavidsonDataset(model_name=self.MODEL_NAME, fixed_set="val")
        elif self.dataset in "wich":
            train_dataset = WichDataset(model_name=self.MODEL_NAME, fixed_set="train")
            dev_dataset = WichDataset(model_name=self.MODEL_NAME, fixed_set="val")
        training_args = TrainingArguments(
            output_dir='../results',          # output directory
            num_train_epochs=2,              # total number of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=128,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='../logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            dataloader_num_workers=6

        )

        self.model.train()
        trainer = WRTrainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=dev_dataset


        )

        trainer.train()
        self.model.save_pretrained('../../models/DistilBERT-'+self.dataset +'.model')

class WRTrainer(tff.Trainer):
    """
    wrapper for a weighted random sampler trainloader.
    """
    def get_train_dataloader(self):
        train_sampler = WeightedRandomSampler(self.train_dataset, 4500)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )