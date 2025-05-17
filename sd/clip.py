import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab:int,n_embd:int, n_tokens:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens,n_embd))            # In original transformer positional embeddings are sinusoidal functions but in CLIP learnt parameters are used for positional encodings


    def forward(self,tokens):
        #(Batch_Size,Seq_Len) -> (Batch_Size,Seq_Len,Dim)  
        x = self.token_embedding(tokens)
        x+=self.position_embedding

        return x
    
class CLIPLayer(nn.Module):
    
    def __init__(self,n_head:int,n_embd:int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head,n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd,4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd,n_embd)

    def forward(self,x:torch.Tensor)->torch.Tensor:

        #(Batch_Size,Seq_Len,Dim)

        residue = x

        ##SELF ATTENTION
        x = self.layernorm_1(x)
        x = self.attention(x,causal_mask=True)
        x+= residue

        ##FEED-FORWARD LAYER

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x*torch.sigmoid(1.702*x)        # QuickGELU activation function (no justification just found better in practice)

        x = self.linear_2(x)

        x += residue

        return x



class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408,768,77)        #Vocab Size, Max Seq_Len, Padding

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)            # 12-> Heads in Multi head Attention 768-> Seq_Len 12->no of layers
        ])

        self.layernorm = nn.LayerNorm(768)

    
    def forward(self,tokens:torch.LongTensor) -> torch.FloatTensor:      #LongTensor because input ids'  are actually  numbers that specifies position of each token inside the vocabulary
        
        tokens = tokens.type(torch.long)

        #(Batch,Seq_Len) -> (Batch_Size,Seq_Len,Dim)                    Dim -> 768

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size,Seq_Len,Dim)
        output = self.layernorm(state)

        return output