import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import math

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()

class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemproalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout = .1):
        super(TemproalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,hidden_size=in_dim,num_layers=layers,dropout=dropout)

    def forward(self, input):
        ori_shape = input.shape
        x = input.permute(3, 0, 2, 1)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
        x,_ = self.rnn(x)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
        x = x.permute(1, 3, 2, 0)
        return x

class TrafficTransformer(nn.Module):
    def __init__(self,in_dim,layers=1,dropout=.1,heads=8):
        super().__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim,dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        self.trans = nn.Transformer(in_dim, heads, layers, layers, in_dim*4, dropout=dropout)

    def forward(self,input, mask):
        x = input.permute(1,0,2)
        x = self.pos(x)
        x = self.lpos(x)
        x = self.trans(x,x,tgt_mask=mask)
        # x = self.trans(x,x)
        return x.permute(1,0,2)

    def _gen_mask(self,input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask

class TRANS(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
    # def __init__(self, dropout=0.1, supports=None, in_dim=2, linearout_dim=12, self.nhid=32, layers=6):
        
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 2)
        super().__init__(config, data_feature)

        self.dropout = config.get('dropout', 0.1)
        
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.randomadj = config.get('randomadj', True)
        self.aptonly = config.get('aptonly', True)
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.linearout_dim=12
        self.device = config.get('device', torch.device('cpu'))

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.nhid,
                                    kernel_size=(1, 1))
        self.start_embedding = TemproalEmbedding(self.nhid, layers=3, dropout=self.dropout)
        self.end_conv = nn.Linear(self.nhid, self.linearout_dim)
        
        self.network = TrafficTransformer(in_dim=self.nhid, layers=6, dropout=self.dropout)
        self.cal_adj(self.adjtype)

        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]

        mask0 = self.supports[0].detach()
        mask1 = self.supports[1].detach()
        mask = mask0 + mask1
        out = 0
        for i in range(1, 7):
            out += mask ** i
        self.mask = out == 0

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

    def forward(self, batch):
        input = batch['X']
        input = input.transpose(1, 3)
        x = self.start_conv(input)
        x = self.start_embedding(x)[..., -1]
        x = x.transpose(1, 2)
        x = self.network(x, self.mask)
        x = self.end_conv(x)
        return x.transpose(1,2).unsqueeze(-1)
    
    def masked_mae(preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted= self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true,0.0)
    
    def predict(self, batch):
        return self.forward(batch)

