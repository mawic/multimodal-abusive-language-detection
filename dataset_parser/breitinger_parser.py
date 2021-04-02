# TODO: Verify that data set copied from Mariams src doesn't differ from the actual MongoDB
# The tweets are tokenized using the DistilBertTokenizer.
# Code adapted from https://huggingface.co/transformers/custom_datasets.html
import os
import pickle
import re

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


class WichDataset(torch.utils.data.Dataset):
    def __init__(self, model_name: str = 'distilbert-base-german-cased', fixed_set = None):
        """
        This class will provide a pytorch data set of UserID, tweet, sentiment (hate/offensive/neither).
        :param model_name: Model name of the huggingface model used for tokenization
        """
        if fixed_set:
            print ("Loading Wich " + fixed_set +" set from fixed split.")
            storagedict = torch.load("../../data/wich/fixed_split.pt")
            tweets = storagedict[fixed_set + "_tweets" ]
            users  = storagedict[fixed_set + "_users" ]
            user_mapping = torch.load("../../data/wich/user_id_mapping.pt")
            labels = storagedict[fixed_set + "_labels" ]
            temp = []
            for x in tweets:
                temp.append(re.sub('@[^\s]+', '@', x)) # remove word after @ in tweet (= username)
            tweets  = temp
        else:
            annotated_tweets = torch.load("../../data/wich/all_tweets_final.pt")
            tweets = annotated_tweets["tweet"]
            labels = []
            for i,x in enumerate(annotated_tweets["label"]):
                if "OTHER" in x:
                    labels.append(0) # Other class
                else:
                    labels.append(1) # Offensive class
            users = torch.load("../../data/wich/user_id_mapping.pt")  # convert user values to int, there's probably some better way
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        encodings_all = tokenizer(tweets, return_tensors='pt', padding=True, truncation=True)
        self.encodings = encodings_all
        self.users = users
        self.labels = labels
        print("Successfully loaded wich dataset.")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        item['userid'] = self.users[idx]
        return item

    def get_tweets_for_user_id(self, user_id):
        idx_list = [i for i,val in enumerate(self.users) if val==user_id]
        for idx in idx_list:
            res = self.__getitem__(idx)
            res['input_ids'] = res['input_ids'].unsqueeze(0)
            res['attention_mask'] = res['attention_mask'].unsqueeze(0)
            res['label'] = res['label'].unsqueeze(0)
            res['userid'] = torch.as_tensor(res['userid']).unsqueeze(0)
            yield res

    def __len__(self):
        return len(self.labels)