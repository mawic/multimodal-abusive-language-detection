import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer


class TF_IDF_TweetHistory(nn.Module):
	def __init__(self, random_subset_size: int = None, vocab_size=500, dataset="davidson"):
		"""
		param: random_subset_offensive_size: number of offensive tweets used to generate most common words
		param: vocab_size: size of vocabulary to be used for Tweet History
		param: dataset: name of the data set, either davidson, waseem or wich.
		"""
		super().__init__()
		self.dataset = dataset
		if "davidson" in dataset:
			with open('../../data/davidson/annotated_tweets_preprocessed', 'rb') as path_tweets:
				annotated_tweets = pickle.load(path_tweets)
			full_length_offensive = 11800  # applies to Davidson data set - refers to offensive tweet count
		elif "waseem" in dataset:
			full_length_offensive = 11501  # applies to Waseem data set - refers to neither tweet count
			annotated_tweets = torch.load("../../data/waseem/annotated_tweets_preprocessed.pt")
		elif "wich" in dataset:
			full_length_offensive = 42238
			annotated_tweets = pd.DataFrame.from_dict(torch.load("../../data/wich/annotated_tweets.pt"))
		self.VOCAB_SIZE = vocab_size
		df = annotated_tweets.copy()

		tweets_class0 = df[df['class'] == 0]['text_preprocessed']
		tweets_class1 = df[df['class'] == 1]['text_preprocessed']
		try:
			tweets_class2 = df[df['class'] == 2]['text_preprocessed']
		except:
			print("Could not load class 2 tweets, probably because dataset is wich?!")
		# adjustment of tweets_class1 necessary because it is too large e.g. for Wich
		# idea: randomly sample a smaller set, determined by random_subset_offensive variables
		if random_subset_size is not None:
			p = random_subset_size / full_length_offensive
			idx = np.random.binomial(1, p, full_length_offensive)
			if "davidson" in dataset:
				tweets_class1 = tweets_class1[idx == 1]
			elif "waseem" in dataset:
				tweets_class2 = tweets_class2[idx == 1]
			elif "wich" in dataset:
				tweets_class0 = tweets_class0[idx == 1]
				p = random_subset_size / len(tweets_class1)
				idx = np.random.binomial(1, p, len(tweets_class1))
				tweets_class1 = tweets_class1[idx == 1]
		if "davidson" in dataset:
			tweets_class2 = df[df['class'] == 2]['text_preprocessed']
			vocab_hate = self.get_mostfrequent_per_class(tweets_class0, 500)
			print('hate done')
			vocab_offensive = self.get_mostfrequent_per_class(tweets_class1, 500)
			print('offensive done')
			vocab_neither = self.get_mostfrequent_per_class(tweets_class2, 500)
			print('neither done')
		elif "waseem" in dataset:
			# keep variable names for now, so that it works as in the original davidson documentation
			vocab_hate = self.get_mostfrequent_per_class(tweets_class0, 500)
			print('neither done')
			vocab_offensive = self.get_mostfrequent_per_class(tweets_class1, 500)
			print('racism done')
			vocab_neither = self.get_mostfrequent_per_class(tweets_class2, 500)
			print('sexism done')
		elif "wich" in dataset:
			# keep variable names for now, so that it works as in the original davidson documentation
			vocab_neither = self.get_mostfrequent_per_class(tweets_class0, 500)
			print('other done')
			vocab_offensive = self.get_mostfrequent_per_class(tweets_class1, 500)
			print('offensive done')
			vocab_hate = None
		self.VOCABULARY = self.get_vocab_collection(vocab_hate, vocab_neither, vocab_offensive)
		self.vocab_per_tweet = self.get_vocab_on_tweet_level(df)
		self.dict_vocab_per_user = self.get_vocab_per_unique_user()

	def get_mostfrequent_per_class(self, tweet_data, nlen=500):
		list_help = []
		for i in tweet_data:
			list_help.append(i)
		tweet_list = []
		for i in range(len(list_help)):
			try:
				if list_help[i][0] == ' ':
					helper = ''
				else:
					helper = list_help[i][0]
			except:
				print(i)
				print(list_help[i])
				continue
			for j in range(1, len(list_help[i])):
				helper = helper + ' ' + list_help[i][j]
			tweet_list.append(helper)

		vectorizer = TfidfVectorizer()
		vectors = vectorizer.fit_transform(tweet_list)
		feature_names = vectorizer.get_feature_names()
		dense = vectors.todense()
		denselist = dense.tolist()
		vocab = pd.DataFrame(denselist, columns=feature_names)

		vocab = vocab.mean().sort_values(ascending=False)
		return vocab[:nlen]

	def get_vocab_collection(self, vocab_hate, vocab_neither, vocab_offensive):
		vocabulary_length = self.VOCAB_SIZE
		list_output = []
		count = 0
		iterator = 0
		ignore_words = ['user', 'pron', 'hashtag', 'rt', 'repeat', 'allcap', 'allcaps', 'number', 'url']
		while count < vocabulary_length:
			if not "wich" in self.dataset:
				if vocab_hate.index[iterator] not in ignore_words:
					list_output.append(vocab_hate.index[iterator])
					ignore_words.append(vocab_hate.index[iterator])
					count += 1
					if count == vocabulary_length:
						break
			if vocab_offensive.index[iterator] not in ignore_words:
				list_output.append(vocab_offensive.index[iterator])
				ignore_words.append(vocab_offensive.index[iterator])
				count += 1
				if count == vocabulary_length:
					break
			if vocab_neither.index[iterator] not in ignore_words:
				list_output.append(vocab_neither.index[iterator])
				ignore_words.append(vocab_neither.index[iterator])
				count += 1
			iterator += 1
		return list_output

	def get_vocab_on_tweet_level(self, df):
		VOCABULARY = self.VOCABULARY
		count_tweets = len(df)
		df_preproc = df['text_preprocessed']
		df_tweet = df['text'].to_numpy().reshape((count_tweets, 1))
		df_userid = df['user'].to_numpy().reshape((count_tweets, 1))
		vocab_per_tweet = np.zeros(shape=(count_tweets, len(VOCABULARY)), dtype=int)

		for i in range(count_tweets):
			for j in range(len(df_preproc[i])):
				help_string = df_preproc[i][j]
				if help_string in VOCABULARY:
					index_help = VOCABULARY.index(help_string)
					vocab_per_tweet[i][index_help] = 1

		for user_id in df['user'].unique():
			index_list = df[df['user'] == user_id].index.tolist()
			user_vector_array = vocab_per_tweet[index_list]
			for index in index_list:
				vocab_per_tweet[index] = np.logical_or.reduce(user_vector_array)

		vocab_per_tweet = np.concatenate((df_tweet, df_userid, vocab_per_tweet), axis=1)
		return vocab_per_tweet

	def check_BOW_encoding(self, user_id, df):
		VOCAB_SIZE = self.VOCAB_SIZE
		VOCABULARY = self.VOCABULARY
		vocab_per_tweet = self.vocab_per_tweet
		df_preproc = df['text_preprocessed']
		index_list = df[df['user'] == user_id].index.tolist()
		corpus = []
		for i in range(len(index_list)):
			j = index_list[i]
			for k in range(len(df_preproc[j])):
				help_string = df_preproc[j][k]
				if help_string not in corpus:
					corpus.append(help_string)
		checker = np.zeros(VOCAB_SIZE)
		for i in range(len(corpus)):
			if corpus[i] in VOCABULARY:
				index_helper = VOCABULARY.index(corpus[i])
				checker[index_helper] = 1
		output = True
		for i in range(len(index_list)):
			if all(checker == vocab_per_tweet[index_list[i]][2:]) != True:
				output = False
		return output

	def get_vocab_per_unique_user(self):
		vocab_per_tweet = self.vocab_per_tweet
		unique_users = np.unique(vocab_per_tweet[:, 1])
		user_count = len(unique_users)
		dict_vocab_per_user = {}
		for (i, user_id) in zip(range(user_count), unique_users):
			index_list = np.argwhere(user_id == vocab_per_tweet[:, 1])
			if "waseem" in self.dataset or "wich" in self.dataset:
				dict_vocab_per_user[str(user_id)] = vocab_per_tweet[index_list][0].reshape(-1)[
				                                    2:]  # different keys, str for davidson, originally int for waseem
			else:
				dict_vocab_per_user[user_id] = vocab_per_tweet[index_list][0].reshape(-1)[2:]
		return dict_vocab_per_user

	def forward(self, user_id):
		"""
		:param user_id: user id for vocab return
		:return: vocab vectors (stacked for batch)
		"""
		res = []
		for x in user_id:
			res.append(torch.from_numpy(np.asarray(self.dict_vocab_per_user[str(x.item())],
			                                       dtype=np.float32)))  # little bit odd implementation due to user id having different types
		return torch.vstack(res) # create batch

	def get_strings_for_shap_visualisations(self, vocab_indices):
		# returns the words belonging to the indices from the shap computations
		VOCABULARY = self.VOCABULARY
		VOCAB_SIZE = self.VOCAB_SIZE
		output_list = []
		index_list = list(range(VOCAB_SIZE))
		vocab_dictionary = dict(zip(index_list, VOCABULARY))
		for i in vocab_indices:
			i = i[1].item()
			output_list.append(vocab_dictionary[i])
		return output_list
