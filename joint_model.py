# TODO: unclear how to split the dataset into 3 sets (train, dev, test) for each separate model

import torch
import torch.nn as nn
from TweetNetwork.graphsage import GraphSAGE
from TweetClassifier.distilbert_sequence_classifier import TweetBERT
from TweetHistory.tf_idf_vectors import TF_IDF_TweetHistory


class JointModel(nn.Module):
    def __init__(self, graph_emb_dim=32, vocab_size=500, random_subset_size=None, dataset = "davidson", freezeBERT = True):
        """
        initalize joint model consisting of BERT, GraphSAGE and BOW (TFIDF implementation)
        :param graph_emb_dim: embedding dimension of the pretrained tweet network GraphSAGE embeddings
        :param vocab_size: length of the BOW vector, most used words
        :param random_subset_size: amount of offensive tweets used to generate most common words
        """
        super().__init__()
        self.device = torch.cuda.is_available()
        self.graph_emb_dim = graph_emb_dim
        self.vocab_size = vocab_size
        self.SAGE = GraphSAGE(num_node_features=2, embedding_size=graph_emb_dim, num_layers=3, trained = True, dataset=dataset)
        self.dataset = dataset
        print ("Successfully initialized TweetNetwork submodel")
        self.BERT = TweetBERT(trained=True, freeze_weights = freezeBERT, dataset=dataset)
        print ("Successfully initialized TweetClassifier submodel")
        self.BOW = TF_IDF_TweetHistory(random_subset_size= random_subset_size, vocab_size = vocab_size, dataset=dataset)
        print ("Successfully initialized TweetHistory submodel")
        if "wich" in dataset:
            self.simple_linear = nn.Linear(graph_emb_dim + 2 + vocab_size, 2)
        else:
            self.simple_linear = nn.Linear(graph_emb_dim + 3 + vocab_size, 3)
        print ("Successfully initialized last final classification layer")


    def forward(self, tweet_input_ids, attention_mask,  user_id, shap = False, vocab_as_one = True, set_Tweet_to_NULL = False, set_BOW_to_NULL = False, set_GRAPH_to_NULL = False, adjusted_user_vocab = None, nodes_to_delete_in_SHAP = None):
        """
        :param tweet_input_ids: the tweet itself, as given by te davidson_parser
        :param attention_mask: Which parts of the inputs should be attended over#TODO
        :param user_id: user id, required for SAGE and BOW parts
        :param masking: "SAGE" or "BOW", String, specifies if SAGE or BOW should be masked.
        :return: result of classifier (simple linear) trained on the (static) output of all three models
        """
        x1 = self.BERT(input_ids = tweet_input_ids, attention_mask=attention_mask)
        #x1.squeeze_() # removes batch dimension from bert output
        x2 = self.SAGE(user_id, nodes_to_delete_in_SHAP) # batch dim has been removed in forward pass of SAGE model
        #x2 = torch.zeros_like(x2)
        x3 = self.BOW(user_id) # doesnt return batch dim
        #x3 = torch.zeros_like(x3)
        if shap == True:
            # Graph adjustments for SHAP
            if set_Tweet_to_NULL == True:
                if self.dataset == "wich":
                    x1 = torch.zeros(tweet_input_ids.shape[0],2)
                else:
                    x1 = torch.zeros(tweet_input_ids.shape[0],3)
            if set_GRAPH_to_NULL == True:
                x2 = torch.zeros(tweet_input_ids.shape[0],self.graph_emb_dim)
            # BOW adjustments for SHAP
            if vocab_as_one == True: # treat BOW as a single feature
                if set_BOW_to_NULL == True:
                    x3 = torch.zeros(tweet_input_ids.shape[0], self.vocab_size)
            else: # treat each element in users vocab as individual element
                x3 = adjusted_user_vocab
        if "cpu" not in str(self.device):
            x1, x2, x3 = x1.to(torch.device("cuda")), x2.to(torch.device("cuda")), x3.to(torch.device("cuda"))
        multi_input_cat = torch.cat([x1,x2,x3], dim = 1)
        res = self.simple_linear(multi_input_cat)
        return res
    

