# TODO: Verify that data set copied from Mariams src doesn't differ from the actual MongoDB
# The tweets are tokenized using the DistilBertTokenizer.
# Code adapted from https://huggingface.co/transformers/custom_datasets.html
import os
import pickle
import re

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


class WaseemDataset(torch.utils.data.Dataset):
    def __init__(self, model_name: str = 'distilbert-base-uncased', fixed_set = None):
        """
        This class will provide a pytorch data set of UserID, tweet, sentiment (hate/offensive/neither).
        :param model_name: Model name of the huggingface model used for tokenization
        """
        if fixed_set:
            print ("Loading Waseem " + fixed_set +" set from fixed split.")
            storagedict = torch.load("../../data/waseem/fixed_split.pt")
            tweets = storagedict[fixed_set + "_tweets" ]
            users  = storagedict[fixed_set + "_users" ]
            labels = storagedict[fixed_set + "_labels" ]
        else:
            with open('../../data/waseem/annotated_tweets', 'rb') as file_tweets:
                annotated_tweets = torch.load(file_tweets)
            tweets = annotated_tweets["text"].values.tolist()
            labels = annotated_tweets["class"].values.tolist()
            users = annotated_tweets["user"].values.tolist()
            users = list(map(int, users))  # convert user values to int, there's probably some better way
        temp = []
        for x in tweets:
            temp.append(re.sub('@[^\s]+', '@', x))  # remove word after @ in tweet (= username)
        tweets = temp
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        encodings_all = tokenizer(tweets, return_tensors='pt', padding=True, truncation=True)
        self.encodings = encodings_all
        self.users = users
        self.labels = labels
        print("Successfully loaded waseem dataset.")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        item['userid'] = self.users[idx]
        return item

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def split_origin_set():
        """
        splits original full length data set in train, dev, test sets and stores it in data
        :return:
        """
        from sklearn.model_selection import train_test_split
        print (os.getcwd())
        annotated_tweets = torch.load('../../data/waseem/annotated_tweets.pt', 'rb')
        #with open('../../data/davidson/follow_relationships', 'rb') as file_followers:
        #    follow_relationships = pickle.load(file_followers)
        labels = annotated_tweets["labels"]
        tweet_ids = annotated_tweets["tweet_ids"]
        users = annotated_tweets["user_ids"]
        tweets = annotated_tweets["texts"]
        s = dict()
        s["train_tweets"], other_tweets, s["train_labels"], other_labels, s[
            "train_users"], other_users, s["train_tweet_ids"], other_ids = train_test_split(tweets, labels, users,tweet_ids, stratify=labels, test_size=0.4)
        s["val_tweets"], s["test_tweets"], s["val_labels"], s["test_labels"], s["val_users"], s[
            "test_users"], s["val_tweet_ids"], s["test_tweet_ids"] = train_test_split(other_tweets, other_labels, other_users, other_ids, stratify=other_labels, test_size=0.5)
        torch.save(s, '../../data/waseem/fixed_split.pt')