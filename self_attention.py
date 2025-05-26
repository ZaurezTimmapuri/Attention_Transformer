import torch
import torch.nn as nn #for module and linear classes used in building model
import torch.nn.functional as F #to use the softmax function for calculating softmax in attention

class SelfAttention(nn.Module): #nn.module is inherited where the nn.module is the base class for all nn modules
    def __init__(self,d_model = 3, # d_model is the dimension of weight and encoding value matrix
              row_dim = 0,  # these are some convience parameters
              col_dim = 1):
        super().__init__()

        self.Weight_Q = nn.Linear(in_features = d_model,
                              out_features = d_model,  #infeatures and outfeatures defines the number of rows and cloums in weight matrix
                              bias = False)
        self.Weight_K = nn.Linear(in_features= d_model,
                              out_features= d_model,
                              # infeatures and outfeatures defines the number of rows and cloums in weight matrix
                              bias=False)
        self.Weight_V = nn.Linear(in_features=d_model,
                              out_features=d_model,
                              # infeatures and outfeatures defines the number of rows and cloums in weight matrix
                              bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,token_encodings):

        Q = self.Weight_Q(token_encodings)  #multiplying weighted q with Q from encoding
        K = self.Weight_K(token_encodings)
        V = self.Weight_V(token_encodings)

        similarity = torch.matmul(Q,K.transpose(dim0=self.row_dim, #applying formula for QK^T
                                                dim1=self.col_dim))

        scaled_similarity = similarity / torch.tensor(K.size(self.col_dim)**0.5)  # applying formula QK^T/dim_k^1/2
        attention_percents = F.softmax(scaled_similarity, dim=self.col_dim)     #applying formula softmax(QK^T/dim_k^1/2)
        attention_score = torch.matmul(attention_percents,V)    #applying formula softmax(QK^T/dim_k^1/2)*V

        return attention_score

encoding_matrix = torch.tensor([[1.16, 0.23,0.12],[0.57, 1.36,0.22],[4.41,-2.16,0.15]])
torch.manual_seed(42)
selfAttention = SelfAttention(d_model= 3, # d_model is the dimension of weight and encoding value matrix
              row_dim = 0,
              col_dim = 1)
results = selfAttention(encoding_matrix)
weighted_Q = selfAttention.Weight_Q.weight.transpose(0,1)

print(results)
print(weighted_Q)
