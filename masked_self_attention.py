import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self,d_model= 2,
                 row_dim=0,
                 col_dim=1):
        super().__init__()

        self.W_Q = nn.Linear(in_features = d_model,
                             out_features = d_model,
                             bias = False)
        self.W_V = nn.Linear(in_features = d_model,
                                 out_features = d_model,
                                 bias = False)
        self.W_K = nn.Linear(in_features = d_model,
                                 out_features = d_model,
                                 bias = False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,token_encodings,mask = None):
        q = self.W_Q(token_encodings)
        v = self.W_V(token_encodings)
        k = self.W_K(token_encodings)

        sims = torch.matmul(q,k.transpose(dim0 = self.row_dim,dim1 = self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None :
            scaled_sims = scaled_sims.masked_fill(mask=mask,value = -1e9)
        attention_percent = F.softmax(scaled_sims,dim=self.col_dim)
        attention_score = torch.matmul(attention_percent,v)
        return attention_score

encodings_matrix = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41,-2.16]])
torch.manual_seed(42)
maskedselfattention = MaskedSelfAttention(d_model = 2,
                                          row_dim = 0,
                                          col_dim = 1)

mask = torch.tril(torch.ones(3,3))
mask = mask == 0

print(mask)
print(maskedselfattention(encodings_matrix,mask))