# TODO: Verify that data set copied from Mariams src doesn't differ from the actual MongoDB
# The tweets are tokenized using the DistilBertTokenizer.
# Code adapted from https://huggingface.co/transformers/custom_datasets.html
import os
import pickle
import re

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


class DavidsonDataset(torch.utils.data.Dataset):
    def __init__(self, model_name: str = 'distilbert-base-uncased', fixed_set = None):
        """
        This class will provide a pytorch data set of UserID, tweet, sentiment (hate/offensive/neither).
        :param model_name: Model name of the huggingface model used for tokenization
        """
        if fixed_set:
            print ("Loading Davidson " + fixed_set +" set from fixed split.")
            storagedict = torch.load("../../data/davidson/fixed_split.pt")
            tweets = storagedict[fixed_set + "_tweets" ]
            users  = storagedict[fixed_set + "_users" ]
            labels = storagedict[fixed_set + "_labels" ]
        else:
            with open('../../data/davidson/annotated_tweets', 'rb') as file_tweets:
                annotated_tweets = pickle.load(file_tweets)
            with open('../../data/davidson/follow_relationships', 'rb') as file_followers:
                follow_relationships = pickle.load(file_followers)
            tweets = annotated_tweets["text"].values.tolist()
            labels = annotated_tweets["class"].values.tolist()
            users = annotated_tweets["user"].values.tolist()
            users = list(map(int, users))  # convert user values to int, there's probably some better way
        temp = []
        for x in tweets:
            l  = re.sub('@[^\s]+', '@', x)  # remove word after @ in tweet (= username)
            #l = re.sub('@ [^\s]+', '@', l)  # remove word after @ in tweet (= username)
            temp.append(l)
        tweets = temp
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        encodings_all = tokenizer(tweets, return_tensors='pt', padding=True, truncation=True)
        self.encodings = encodings_all
        self.users = users
        self.labels = labels
        print("Successfully loaded davidson dataset.")

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

    @staticmethod
    def split_origin_set():
        """
        splits original full length data set in train, dev, test sets and stores it in data
        :return:
        """
        from sklearn.model_selection import train_test_split
        with open('../../data/davidson/annotated_tweets', 'rb') as file_tweets:
            annotated_tweets = pickle.load(file_tweets)
        with open('../../data/davidson/follow_relationships', 'rb') as file_followers:
            follow_relationships = pickle.load(file_followers)
        tweets = annotated_tweets["text"].values.tolist()
        labels = annotated_tweets["class"].values.tolist()
        users = annotated_tweets["user"].values.tolist()
        users = list(map(int, users))
        s = dict()
        s["train_tweets"], other_tweets, s["train_labels"], other_labels, s[
            "train_users"], other_users = train_test_split(tweets, labels, users, stratify=labels, test_size=0.4)
        s["val_tweets"], s["test_tweets"], s["val_labels"], s["test_labels"], s["val_users"], s[
            "test_users"] = train_test_split(other_tweets, other_labels, other_users, stratify=other_labels, test_size=0.5)
        torch.save(s, '../../data/davidson/fixed_split.pt')