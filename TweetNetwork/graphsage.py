import csv
import pickle

import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch_geometric.data import NeighborSampler as RawNeighborSampler


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G, label_attribute='user_id')
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=3,  # consider changing to higher value
                                coalesced=False)[:, 1]  #

        neg_batch = torch.randint(0, self.adj_t.size(0), (batch.numel(),),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, trained: bool = False, bidirectional: bool = False, num_node_features: int = 2,
                 embedding_size=32, num_layers=3, freeze_weights: bool = False, dataset="davidson"):
        """
        Wrapper class for GraphSAGE
        :param trained: use pretrained GraphSAGE model
        :param bidirectional: specify bidirectionally or unidirectionally (pre-)trained model
        :param num_node_features: number of input features per node
        :param embedding_size: final embedding size = size of GraphSAGE hidden layer
        :param num_layers: no of hidden layers of GraphSAGE
        :param freeze_weights: freeze weights of model, used when training the joint model
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAGE(num_node_features, hidden_channels=embedding_size, num_layers=3)
        self.model.to(self.device)
        self.bidirectional = bidirectional
        self.dataset = dataset
        if trained:
            self.model.eval()
            if bidirectional:
                self.user_to_embeddings = torch.load('../../models/GS_bidirect_' + dataset + '.embeds')
                self.edge_index = torch.load('../../models/torch_G_bidirect_' + dataset + '.edge_index')
                self.model.load_state_dict(
                    torch.load('../../models/GS_bidirect_' + dataset + '.model', map_location=self.device))
            else:
                self.user_to_embeddings = torch.load('../../models/GS_' + dataset + '.embeds')
                self.edge_index = torch.load('../../models/torch_G_' + dataset + '.edge_index')
                self.model.load_state_dict(
                    torch.load('../../models/GS_' + dataset + '.model', map_location=self.device))
        if "davidson" in dataset:
            with open('../../data/davidson/Users_no_Network_Davidson.csv', newline='', encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                self.null_user_set = [int(x[0]) for x in reader]
            with open('../../data/davidson/community_graph', 'rb') as path_tweets:
                self.original_graph = pickle.load(path_tweets)
        elif "waseem" in dataset:
            with open('../../data/waseem/Users_no_Network_Waseem.csv', newline='', encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                self.null_user_set = [int(x[0]) for x in reader]
            temp = {}
            for k, v in self.user_to_embeddings.items():
                temp[str(
                    k.item())] = v  # TODO: fix this, both davidson and waseem need the same keys, ideally ints instead of string
            self.user_to_embeddings = temp
            self.original_graph = torch.load('../../data/waseem/community_graph.pt')
        elif "wich" in dataset:
            self.null_user_set = []
            uidmap = torch.load("../../data/wich/user_id_mapping.pt")
            uidmap.set_index("user_id", inplace=True)
            all_uids = set(uidmap.index.values)
            for x in all_uids:
                if str(x) not in self.user_to_embeddings.keys():
                    self.null_user_set.append(x.item())
            # TODO: Probably change dictionary keys to match actual usernames, or use same user ids as in the vocab models
            self.original_graph = torch.load('../../data/wich/community_graph.pt')
        self.null_user_set = set(self.null_user_set)
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, user_id, SHAP_del_users=None):
        zeros = torch.zeros(32)
        res = []
        for x in user_id:
            if x.item() in self.null_user_set:
                res.append(zeros)
            else:
                if SHAP_del_users:
                    # rerun GraphSAGE on network without users in list to obtain altered embedding
                    res.append(self.forward_on_graph_without_nodelist(user_id, SHAP_del_users))
                else:
                    res.append(self.user_to_embeddings[str(x.item())])
        return torch.vstack(res)

    def forward_on_graph_without_nodelist(self, user_id, nodes_to_remove):
        """   
        :param user_id: current user id
        :param nodes_to_remove: nodes that need to be removed from user graph e.g. according to SHAP
        :return: new embedding generated from running the pretrained graphsage model on graph without nodes in nodes_to_remove
        """
        # sanity check: SHAP cannot remove the node that is being observed
        nodes_to_remove = set(nodes_to_remove)
        if user_id in nodes_to_remove:
            raise Exception("current node in SHAP-to-delete-set")
        original_graph = self.original_graph
        node_data = dict(original_graph.nodes(data=True))
        features_df = pd.DataFrame.from_dict(node_data, orient='index')
        if self.bidirectional:
            raise NotImplementedError("bidirectional (with SHAP) has not been implemented")
            # G = nx.DiGraph()
            #
            # for useridx, user in users.iterrows():
            #     G.add_node(useridx, hatespeech=user["hatespeech"], offensive=user["offensive"], neither=user["neither"],
            #                user_class=user["user_class"])
            # G.add_edges_from(follow_relationships.to_numpy())
            # print('COMMUNITY GRAPH - Bidirectional')
            # print('Number of nodes: ', G.number_of_nodes(), ' Number of edges: ', G.number_of_edges())
        else:
            G = nx.Graph(original_graph, node_features=features_df['user_class'])
            print (len(G))
            G.remove_nodes_from(nodes_to_remove)
            print (len(G))
            # print('COMMUNITY GRAPH - Undirectional')
            # print('Number of nodes: ', G.number_of_nodes(), ' Number of edges: ', G.number_of_edges())
        torch_G = from_networkx(G)
        helper = torch.ones(len(torch_G.user_class))
        x = torch.stack((torch_G.user_class, helper), 1)
        x = x.type(torch.FloatTensor)
        x, edge_index = x.to(self.device), torch_G.edge_index.to(self.device)
        node_embeddings = self.model.full_forward(x, edge_index)
        for i, n_id in enumerate(torch_G.user_id):
            if n_id == str(user_id.item()):
                return node_embeddings[i]
        raise Exception("User ID not in node_embs")