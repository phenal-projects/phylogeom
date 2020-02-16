import torch
from torch import nn
from torch.nn import functional as func
from torch_geometric import nn as gnn


class TreeSupport(nn.Module):
    def __init__(self, feat_in, hidden_features=10):
        """
        A class for graph nodes classification. Contains two convolution layers
        :param feat_in: Number of features of one node
        :param hidden_features: Number of features between layers
        """
        super(TreeSupport, self).__init__()
        self.conv1 = gnn.GINConv(
            nn.Sequential(nn.Linear(feat_in, hidden_features), nn.Linear(hidden_features, hidden_features)),
            train_eps=True
        )
        self.conv2 = gnn.GINConv(
            nn.Sequential(nn.Linear(hidden_features, max(hidden_features // 2, 1)),
                          nn.Linear(max(hidden_features // 2, 1), max(hidden_features // 2, 1))),
            train_eps=True
        )
        self.conv3 = gnn.SAGEConv(max(hidden_features // 2, 1), 1)

    def forward(self, data):
        """
        Forward propagation
        :param data:
            x: (Nodes x Features)
            edge_index: (2 x Edges)
            edge_attr: (Edges x 1)
        :return: results of the model (Nodes x 1)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.squeeze()

        x = self.conv1(x, edge_index)
        x = func.relu(x)
        x = func.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = func.relu(x)
        x = func.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return func.relu(x)


class _Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        """
        An encoder for seq AutoEncoder
        :param input_size: input feature size
        :param hidden_size: size of the hidden layer
        :param num_layers: number of layers in LSTM
        """
        super(_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=1.4)
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=1.4)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()  # CHANGE THIS, if training on cpu
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class _Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers):
        """
        An encoder for seq AutoEncoder
        :param output_size: output (actually, input) feature size
        :param hidden_size: size of the hidden layer
        :param num_layers: number of layers in LSTM
        """
        super(_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True, dropout=0.2)

        # initialize weights
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=1.4)
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=1.4)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).cuda()

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class GappedSeq2Vec(nn.Module):
    def __init__(self, res_emb_dim=7, hidden_size=100, num_layers=2):
        """
        Autoencoder for gapped protein sequences
        :param res_emb_dim: Number of dimensions of residue embeddings
        :param hidden_size: Middle representation size
        :param num_layers: layers of LSTM in en(de)coders
        """
        super(GappedSeq2Vec, self).__init__()
        self.embedd = nn.Embedding(22, res_emb_dim)
        self.encoder = _Encoder(res_emb_dim, hidden_size, num_layers)
        self.decoder = _Decoder(hidden_size, res_emb_dim, num_layers)

    def forward(self, x):
        x = self.embedd(x)
        encoded_x = self.encoder(x).expand(-1, x.shape[1], -1)
        decoded_x = self.decoder(encoded_x)
        return decoded_x, x.detach()


class AlignSAGE(nn.Module):
    def __init__(self, feat_in, n_classes=3, max_nodes=45):
        """
        Initializes class for alignment classification
        :param feat_in: the second dimension size of node feature matrix
        :param n_classes: the number of classes in dataset
        :param max_nodes: maximum number of nodes in a graph
        """
        super(AlignSAGE, self).__init__()
        first_hidden = max(feat_in // 2, 1)
        second_hidden = max(feat_in // 4, 1)
        self.N = max_nodes
        self.sage1 = gnn.SAGEConv(feat_in, first_hidden)
        self.pooling1 = gnn.SAGPooling(first_hidden)
        self.sage2 = gnn.SAGEConv(first_hidden, second_hidden)
        self.pooling2 = gnn.SAGPooling(second_hidden)
        self.lin1 = nn.Linear(10, 100)
        self.lin2 = nn.Linear(100, n_classes)

    def forward(self, x, edge_index, edge_weights, batch, debug=False):
        """
        :param x: nodes' features (Nodes x Features_In)
        :param edge_index: COO-formatted sparse graph edges(2 x Edges)
        :param edge_weights: the weights of corresponding edges
        :param batch: actually just a graph labels. Indicates, which data belongs to the specific graph in the batch.
            See pytorch geometric docs.
        :param debug: whether to print shapes after each layer or not
        :return: a tensor containing labels for every graph in the batch
        """
        if debug:
            print(x.shape)
        x = self.sage1(x, edge_index, edge_weights)  # (NxFI, NxN) --> (NxFO)
        if debug:
            print(x.shape)
        x, edge_index, edge_weights, batch, _, _ = self.pooling1(x, edge_index, edge_weights, batch)
        if debug:
            print(x.shape)
        x = func.dropout(x, training=self.training)
        if debug:
            print(x.shape)
        x = self.sage2(x, edge_index, edge_weights)
        if debug:
            print(x.shape)
        x, edge_index, edge_weights, batch, _, _ = self.pooling2(x, edge_index, edge_weights, batch)
        x = func.dropout(x, training=self.training)
        if debug:
            print(x.shape)
        x = gnn.global_sort_pool(x, batch, 10)
        if debug:
            print(x.shape)
        x = func.relu(self.lin1(x))
        if debug:
            print(x.shape)
        x = func.relu(self.lin2(x))
        if debug:
            print(x.shape)
        return x
