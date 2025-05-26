import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention (nn.Module):
    def __init__(self,d_model=2,row_dim = 0,col_dim = 1):
        super().__init__()

        self.w_q = nn.Linear(in_features = d_model,
                             out_features = d_model,
                             bias = False)
        self.w_k = nn.Linear(in_features = d_model,
                             out_features = d_model,
                             bias = False)
        self.w_v = nn.Linear(in_features = d_model,
                             out_features = d_model,
                             bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,encoding_q,encoding_k,encoding_v,mask = None):
        q = self.w_q(encoding_q)
        k = self.w_k(encoding_k)
        v = self.w_v(encoding_v)

        sim = torch.matmul(q,k.transpose(dim0 = self.row_dim,dim1 = self.col_dim))
        scaled_sim = sim / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None :
            scaled_sim = scaled_sim.masked_fill(mask = mask,value = -1e9)
        attention_percent = F.softmax(scaled_sim , dim = self.col_dim)
        attention_score = torch.matmul(attention_percent,v)
        return attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model = 2, row_dim = 0, col_dim = 1, num_heads = 1):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model, row_dim, col_dim) for _ in range(num_heads)])
        self.col_dim = col_dim

    def forward(self,encoding_q,encoding_k,encoding_v,mask=None):
        return torch.cat([head(encoding_q,encoding_k,encoding_v,mask) for head in self.heads], dim = self.col_dim)


encoding_q = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41,-2.16]])
encoding_k = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41,-2.16]])
encoding_v = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41,-2.16]])

mask = torch.tril(torch.ones(3,3))
mask = mask == 0

torch.manual_seed(42)
attention = Attention(d_model = 2, row_dim = 0 , col_dim = 1)
print(attention(encoding_q,encoding_k,encoding_v,mask))


torch.manual_seed(42)
multiheadattention = MultiHeadAttention(d_model = 2, row_dim = 0, col_dim = 1, num_heads = 1)
print(multiheadattention(encoding_q,encoding_k,encoding_v,mask))