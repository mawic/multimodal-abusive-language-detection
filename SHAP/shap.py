# Approximate Shapley values
import pickle

import shap
import torch
from torch.distributions import Binomial
import transformers
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import shap.plots as splt
from matplotlib.ticker import FuncFormatter

class ShapExplainer():
    def __init__(self, model_to_explain, tweet_as_one = False, vocab_as_one=True, network_as_one = True, dataset = "davidson", untokenize = True):  
              
        super().__init__()
        self.model_to_explain = model_to_explain
        self.APPROX_LEVEL = 15
        self.P = 0.7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tweet_as_one = tweet_as_one
        self.vocab_as_one = vocab_as_one
        self.network_as_one = network_as_one
        self.dataset = dataset
        if dataset == 'wich':
            self.classes = 2
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
        else:
            self.classes = 3
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.untokenize = untokenize
        if network_as_one == False:
            if dataset == "davidson":
                self.community_dictionary = torch.load('../../data/davidson/community_dictionary_user_Davidson.pt')
                self.cluster_dict_main_class = torch.load('../../data/davidson/communities_main_class.pt')
            elif dataset == "waseem":
                self.community_dictionary = torch.load('../../data/waseem/community_dictionary_user_Waseem.pt')
                self.cluster_dict_main_class = torch.load('../../data/waseem/communities_main_class.pt')
            elif dataset == "wich":
                self.community_dictionary = torch.load('../../data/wich/community_dictionary_user_Wich.pt')
                self.cluster_dict_main_class = torch.load('../../data/wich/communities_main_class.pt')
        

    def approximate_shap_values(self, input_ids, attention_mask, user_id):
        """
        main method to compute SHAP values for each feature specified by tweet input and user input
        :param input_ids: from BERT
        :param attention_mask: from BERT
        :param user_id: identifies user specific information
        :return: shapley_values: Matrix containing the shapley values for all features and each class
        :return: max_pred.item(): Class predicted by the model
        :return: [words_in_tweet, vocab_features, network_features]: Feature distribution
        :return: vocab_indices: Indices of the vocabulary of the user in the vocabulary dictionary
        """  
        
        self.model_to_explain.eval()
        prediction = self.model_to_explain(input_ids, attention_mask, user_id)

        prediction = torch.nn.functional.softmax(prediction)
        max_pred = torch.argmax(prediction)
        
        #tweet feature initialisation
        if self.tweet_as_one == False:
            words_in_tweet = attention_mask.sum().item() - 2  # the -2 is for eliminating the CLS and SEP token
        else:
            words_in_tweet = 1
        
        #vocabulary feature initialisation
        if self.vocab_as_one == True:
            vocab_features = 1
            adjusted_user_vocab = None
            vocab_indices = None
        else:
            user_vocab = self.model_to_explain.BOW(user_id)
            vocab_features = user_vocab.sum(dtype = torch.int32).item()
            vocab_indices = torch.nonzero(user_vocab)
            set_BOW_to_NULL = False
        
        #network feature initialisation
        if self.network_as_one == True:
            network_features = 1
        else:
            user_dictionary = self.community_dictionary[str(user_id.item())]
            count_com_relations = len(user_dictionary["communities"])
            if count_com_relations > 0:
                network_features = count_com_relations
            else:
                network_features = 1

        set_Tweet_to_NULL = False

        no_of_features = words_in_tweet + vocab_features + network_features 

        shapley_values = torch.zeros((self.APPROX_LEVEL, no_of_features, self.classes))

        for trial in range(self.APPROX_LEVEL):
            # sampling for feature permutations
            feature_subset = Binomial(1, torch.tensor([self.P]*no_of_features)).sample()

            for feature in range(no_of_features):

                #compute prediction with feature included
                helper = feature_subset[feature].item()
                feature_subset[feature] = 1

                if self.tweet_as_one == False:
                    adjusted_mask = self.subsetting_attention_mask(attention_mask, feature_subset)
                else:
                    adjusted_mask = attention_mask.clone().detach()
                    if feature_subset[0] == 0: 
                        set_Tweet_to_NULL = True
                    else:
                        set_Tweet_to_NULL = False

                if self.vocab_as_one == True:
                    if feature_subset[-(network_features + 1)] == 0:
                        set_BOW_to_NULL = True
                    else:
                        set_BOW_to_NULL = False
                else:
                    adjusted_user_vocab = self.subsetting_user_vocab(user_vocab, vocab_indices, feature_subset, words_in_tweet)

                if self.network_as_one == True:
                    nodes_to_delete_in_SHAP = None
                    if feature_subset[-1] == 0:
                        set_GRAPH_to_NULL = True
                    else:
                        set_GRAPH_to_NULL = False
                else:
                    set_GRAPH_to_NULL = False
                    if count_com_relations > 0:
                        nodes_to_delete_in_SHAP = self.list_user_relationships_to_delete(user_dictionary, feature_subset[-count_com_relations:])
                    else:
                        nodes_to_delete_in_SHAP = None

                prediction_with = self.model_to_explain(input_ids, adjusted_mask, user_id, True, self.vocab_as_one, set_Tweet_to_NULL, set_BOW_to_NULL, set_GRAPH_to_NULL, adjusted_user_vocab, nodes_to_delete_in_SHAP)
                prediction_with = torch.nn.functional.softmax(prediction_with)


                #compute prediction without feature included
                feature_subset[feature] = 0

                if self.tweet_as_one == False:
                    adjusted_mask = self.subsetting_attention_mask(attention_mask, feature_subset)
                else:
                    adjusted_mask = attention_mask.clone().detach()
                    if feature_subset[0] == 0: 
                        set_Tweet_to_NULL = True
                    else:
                        set_Tweet_to_NULL = False

                if self.vocab_as_one == True:
                    if feature_subset[-(network_features + 1)] == 0:
                        set_BOW_to_NULL = True
                    else:
                        set_BOW_to_NULL = False
                else:
                    adjusted_user_vocab = self.subsetting_user_vocab(user_vocab, vocab_indices, feature_subset, words_in_tweet)

                if self.network_as_one == True:
                    nodes_to_delete_in_SHAP = None
                    if feature_subset[-1] == 0:
                        set_GRAPH_to_NULL = True
                    else:
                        set_GRAPH_to_NULL = False
                else:
                    set_GRAPH_to_NULL = False
                    if count_com_relations > 0:
                        nodes_to_delete_in_SHAP = self.list_user_relationships_to_delete(user_dictionary, feature_subset[-count_com_relations:])
                    else:
                        nodes_to_delete_in_SHAP = None

                prediction_without = self.model_to_explain(input_ids, adjusted_mask, user_id, True, self.vocab_as_one, set_Tweet_to_NULL, set_BOW_to_NULL, set_GRAPH_to_NULL, adjusted_user_vocab, nodes_to_delete_in_SHAP)
                prediction_without = torch.nn.functional.softmax(prediction_without)

                #prediction with and without return array of length three (for the respective number of classes)
                shapley_values[trial][feature] = prediction_with - prediction_without

                feature_subset[feature] = helper

        shapley_values = shapley_values.mean(axis=0).T

        return shapley_values, max_pred.item(), [words_in_tweet, vocab_features, network_features], vocab_indices

    def subsetting_attention_mask(self, attention_mask, feature_subset):
        # Masking of individual tokens within tweet based on sampled feature_subset vector - used in approximate_shap_values method
        
        adjusted_mask = attention_mask.clone().detach()

        size_attention_mask = attention_mask.sum().item() - 2
        feature_subset  = feature_subset.to(torch.device("cuda"))
        adjusted_mask[0][1:size_attention_mask+1] = attention_mask[0][1:size_attention_mask+1] * feature_subset[
                                                                                           :size_attention_mask]

        return adjusted_mask

    def subsetting_user_vocab(self, user_vocab, vocab_indices, feature_subset, words_in_tweet):
        
        # Masking of individual tokens within vocabulary based on sampled feature_subset vector - used in approximate_shap_values method
        
        with torch.no_grad():
            adjusted_user_vocab = user_vocab.clone()
            size_vocab = user_vocab.sum(dtype=torch.int32).item()
            reduced_feature_subset = feature_subset[words_in_tweet:words_in_tweet + size_vocab]
            reduced_feature_subset = reduced_feature_subset.to(self.device)
            adjusted_user_vocab = adjusted_user_vocab.to(self.device)
            adjusted_user_vocab.squeeze_()
            vi = vocab_indices[:,1]
            vi = vi.to(self.device)
            adjusted_user_vocab.index_copy_(0, vi, reduced_feature_subset)#[~a] = 0
            adjusted_user_vocab.unsqueeze_(0)
       
        return adjusted_user_vocab

    def list_user_relationships_to_delete(self, user_dictionary, reduced_feature_subset):
        
        # Masking of network connections to all users of a given community with nulled feature tokenin feature_subset vector - used in approximate_shap_values method
        
        nodes_to_delete_in_SHAP = []

        for i, j in enumerate(reduced_feature_subset):
            if j == 0:
                nodes_to_delete_in_SHAP.extend(user_dictionary["related_users"][i])

        return nodes_to_delete_in_SHAP



    def get_visualization_config(self, input_ids, user_id, shapley_values, vocab_indices, feature_distribution):
        
        # Preparation of results from approximate_shap_values method for visualization
        
        _, vocab_features, network_features = feature_distribution

        if self.tweet_as_one == False:
            sv, tokens = self.aggregate_tweet_tokens(shapley_values, input_ids)
            words_in_tweet = len(tokens)
            tweet_as_string = self.tokenizer.convert_tokens_to_string(tokens[:words_in_tweet])
        else:
            words_in_tweet = len(input_ids[input_ids>0])
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens = False)[:words_in_tweet]
            sv = shapley_values.detach().numpy()
            tweet_as_string = self.tokenizer.convert_tokens_to_string(tokens[1:words_in_tweet-1])

        features = tokens

        if self.tweet_as_one == False:
            features = tokens
        else:
            features = ['Tweet']

        if self.vocab_as_one == True:
            features.append('Vocabulary')
        else:
            vocab_list = self.model_to_explain.BOW.get_strings_for_shap_visualisations(vocab_indices)
            for help_string in vocab_list:
                features.append(help_string + '(vocab)')

        if self.network_as_one == True:
            features.append('Network')
        else:

            user_dictionary = self.community_dictionary[str(user_id.item())]
            
            if user_dictionary['communities'] == []:
                features.append('No Network')
            else:
                own_community = user_dictionary["own_community"][0]

                for i in range(network_features):
                    print(i, range(network_features), len(user_dictionary["related_users"]))
                    count_connections = len(user_dictionary["related_users"][i])

                    if user_dictionary["communities"][i] == own_community:
                        main_class_o_c = user_dictionary["main_class_of_community"][0]
                        features.append('Own ('+ str(own_community) + '), class: ' + str(main_class_o_c) + ', ccts: ' + str(count_connections))
                    else:
                        main_class_other = self.cluster_dict_main_class.get(user_dictionary["communities"][i])
                        features.append('Community ' + str(user_dictionary["communities"][i]) + ' (class: ' + str(main_class_other) + ', #connec: ' + str(count_connections) + ")")

        if self.tweet_as_one == False:
            palette = list(sns.color_palette("Spectral", words_in_tweet)) + [(0.5,0.5,0.5)]*(vocab_features) + [(0,0,0)]*(network_features)
        else:
            palette = list([(0.5,0.5,0.5)]*(1) + [(0.5,0.5,0.5)]*(vocab_features) + [(0,0,0)]*(network_features))

        return sv, tweet_as_string, features, palette

    # %%

    def visualize_shap_values(self, input_ids, user_id, tweet_label, shapley_values, predicted_class, feature_distribution, vocab_indices):
        
        # Standard visualization of SHAP results in bar plot
        
        shap_values, tweet_as_string, features, palette = self.get_visualization_config(input_ids, user_id, shapley_values, vocab_indices, feature_distribution)

        print(f'Original tweet: {tweet_as_string}')
        print(f'Features contributions to class {predicted_class} (real class: {tweet_label.item()})')
        
        
        plot_height = len(features) * 0.2
        print(len(features),len(shap_values))

        df = pd.DataFrame({
        'features': features,
        'score': shap_values,
        })

        sns.set(rc={'figure.figsize':(6.5,plot_height)}, style="whitegrid")
        ax = sns.barplot(x=df.score, y=df.index, palette=palette, dodge=False, orient='h')
        ax.set_yticklabels(df.features)
        # ax.set(xlim=(-0.1,0.4))
        plt.show()

    def aggregate_tweet_tokens(self, shap_values, input_ids):
        
        # aggregate tweet tokens to make split words from tokenizer readable and aggregate their SHAP values
     
        words_in_tweet = len(input_ids[input_ids>0])
        
        tweet_as_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens = False)[1:words_in_tweet-1]
     
        token_len = len(tweet_as_tokens)
        sv = shap_values.detach().numpy()
        count_shap_values = len(sv)
        helper_sv = sv[-(count_shap_values-token_len):]
        sv = sv[0:token_len]
        new_tweet_fancy = []
        for x in tweet_as_tokens:
            if "CLS" in x or "SEP" in x:
                #new_tweet_fancy.append('')
                pass
            else:
                if "##" in x:
                    new_tweet_fancy.append(x.strip("##"))
                else:
                    new_tweet_fancy.append(" "+ x)
        if self.untokenize:
            added_values = []
            untokenized_words = []
            grp_idx = []
            cur_grp = 0
            for i, x in enumerate(tweet_as_tokens):
                if "##" not in x:
                    grp_idx.append([])
                    grp_idx[cur_grp].append(i)
                    cur_grp +=1
                else:
                    grp_idx[cur_grp-1].append(i)
            for idxs in grp_idx:
                tempval = 0
                tempstr= ""
                for i in idxs:
                    tempval+= sv[i]
                    x = tweet_as_tokens[i]
                    if "CLS" in x or "SEP" in x:
                        tempstr += ''
                    else:
                        if "##" in x:
                            tempstr += x.strip("##")
                        else:
                            tempstr += " " + x
                untokenized_words.append(tempstr)
                added_values.append(tempval)
            sv = np.array(added_values)
            sv = np.concatenate([sv, helper_sv])
            new_tweet_fancy = untokenized_words
        return sv, new_tweet_fancy

    def visualize_text_plot(self, shap_values, input_ids):
        
        # SHAP force plot

        sv, new_tweet_fancy = self.aggregate_tweet_tokens(shap_values, input_ids)
        sv = sv[0:len(new_tweet_fancy)]
        avg = np.mean(np.sqrt(np.abs(sv))) # this was just chosen empirically for now
        sv = shap.Explanation(values=sv, data=new_tweet_fancy, base_values=0)

        splt.text(sv, cmax=avg)


    def vocab_plot_shap(self, shapley_values, predicted_class, feature_distribution, vocab_indices):        
        
        # Waterfall plot for vocabulary
        
        if self.vocab_as_one == True:
            print("Vocabulary has be split in order to visualize it. Please change vocab_as_one to False!")
            return 0
        
        vocab_list = self.model_to_explain.BOW.get_strings_for_shap_visualisations(vocab_indices)
        voc_shap = shapley_values[predicted_class][feature_distribution[0]:feature_distribution[0]+feature_distribution[1]]
        voc_shap = voc_shap.detach().numpy()

        output = []

        for i in range(len(vocab_list)):
            output.append((vocab_list[i],voc_shap[i]))
        
        def takeSecond(elem):
            return elem[1]
        
        output.sort(key=takeSecond)

        top_words = []
        top_scores = []
        neg_words = []
        neg_scores = []
        cum_rest = 0

        for i in range(len(output)):
            if i <5:
                top_words.append(output[-(i+1)][0])
                top_scores.append(output[-(i+1)][1])
                neg_words.append(output[i][0])
                neg_scores.append(output[i][1])
            elif i >= len(output) - 5:
                pass
            else:
                cum_rest += output[i][1]

        count_rest = len(output) - 10
        words = top_words +  ["REST (" + str(count_rest) +")"] + neg_words
        scores = top_scores + [cum_rest] + neg_scores
        
        def my_ceil(a, precision=0):
            return np.true_divide(np.ceil(a * 10**precision), 10**precision)
        def my_floor(a, precision=0):
            return np.true_divide(np.floor(a * 10**precision), 10**precision)
        
        sum_top_scores = sum(top_scores)
        sum_all_scores = sum(top_scores) + sum(neg_scores) + cum_rest

        upper_limit = max(sum_top_scores, sum_top_scores + cum_rest)
        upper_limit = upper_limit + 1/10 * upper_limit
        upper_limit = my_ceil(upper_limit, 1)

        lower_limit = min(0, sum_all_scores)

        if lower_limit < 0:
            lower_limit = lower_limit + 1/10 * lower_limit
            lower_limit = my_floor(lower_limit, 1)

        index = words
        data = {'amount': scores}


        #Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data=data,index=index)
        blank = trans.amount.cumsum().shift(1).fillna(0)

        #Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["OVERALL"]= total
        blank.loc["OVERALL"] = total

        #The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan

        #When plotting the last element, we want to show the full bar,
        #Set the blank to 0
        blank.loc["OVERALL"] = 0

        #Plot and label
        my_plot = trans.plot(kind='bar', stacked=True, bottom=blank,legend=None, figsize=(12, 5), title="SHAPLEY values for vocabulary")
        my_plot.plot(step.index, step.values)
        my_plot.set_xlabel("Vocabulary")

        #Get the y-axis position for the labels
        y_height = trans.amount.cumsum().shift(1).fillna(0)

        #Get an offset so labels don't sit right on top of the bar
        offset = trans.max()
        neg_offset = offset / 3
        pos_offset = offset / 50
        plot_offset = offset / 15


        #Start label loop
        loop = 0
        for index, row in trans.iterrows():
            # For the last item in the list, we don't want to double count
            if row['amount'] == total:
                y = y_height[loop]
            else:
                y = y_height[loop] + row['amount']
            # Determine if we want a neg or pos offset
            if row['amount'] > 0:
                y += pos_offset
            else:
                y -= neg_offset
            my_plot.annotate("{:.3f}".format(row['amount']),(loop,y),ha="center")
            loop+=1

        #Scale up the y axis so there is room for the labels
        my_plot.set_ylim(lower_limit,upper_limit)
        #Rotate the labels
        my_plot.set_xticklabels(trans.index,rotation=0)
        my_plot.get_figure().savefig("waterfall.png",dpi=200,bbox_inches='tight')

    def vocab_plot_shap(shap_values, tweet_as_string, untokenize = False):
        pass

    def plot_network(self, shap_output):
        """
        run shap calculation with tweet as one, vocab as one = True, network as one = False
        :param shap_output: provide list of shap output like [[i,user_id.item(),shapley_values, res, test_max_pred]]
        :return: takes first element from shap_output list and return visualization
        """
        with open("../../data/davidson/community_graph", "rb") as f:
            communities = pickle.load(f)
        comm_dict = torch.load("../../data/davidson/community_dictionary_user_Davidson.pt")
        com_partition = torch.load("../../data/davidson/community_partition_Davidson.pt")
        def visualize_user_relations(userid, shap_values, pred_class):
            big_dict = dict()
            my_comm = comm_dict[userid]['own_community'][0]
            group = 1
            count_hater_group = dict()
            for x, y in zip(comm_dict[userid]['communities'], comm_dict[userid]['related_users']):
                big_dict[x] = dict()
                for z in y:
                    print (z)
                    if communities.nodes()[z]["user_class"] == pred_class:
                        try:
                            count_hater_group[x] += 1
                        except:
                            count_hater_group[x] = 1
                    else:
                        count_hater_group[x] = 0
                if x == my_comm:
                    big_dict[x]['group'] = 0
                else:
                    big_dict[x]['group'] = group
                    group += 1
                big_dict[x]['direct_relation'] = len(y)
            for x in big_dict.keys():
                count_size = 0
                for k, v in com_partition.items():
                    if v == x:
                        count_size += 1
                big_dict[x]["value"] = 30#count_size  # *0.1
                big_dict[x]["label"] = "{}, {}/{}".format(x, count_hater_group[x], big_dict[x]["direct_relation"])
            G = nx.Graph()
            for i, x in enumerate(big_dict.keys()):
                print (shap_values[pred_class][i].item())
                if shap_values[pred_class][i].item() * 100 <= 0:
                    big_dict[x]['color'] = '#118cfe'
                    print ("nothere")
                else:
                    print (x)
                    print ("here")
                    big_dict[x]['color'] = '#fe004d'

            G.add_nodes_from(big_dict.items())
            G.add_node("user", size=10, shape="ellipse", color='#fffff0')
            for x in big_dict.keys():
                G.add_edge("user", x)
            for i, x in enumerate(G.edges(data=True)):
                try:
                    n1, n2, data = x
                    data["width"] = abs(shap_values[pred_class][i].item()) * 500
                    data["label"] = round(shap_values[pred_class][i].item(), 4)
                except:
                    continue
            print(big_dict)
            return big_dict, my_comm, G
        l = visualize_user_relations(str(shap_output[0][1]), shap_output[0][2][:, 2:].detach().numpy(),
                                     pred_class=shap_output[0][3].item())
        nt = Network(heading="User ID: " + str(shap_output[0][1]) + ", Network Graph")
        nt.from_nx(l[2])
        nt.show("nx.html")


#fe004d" #118cfe