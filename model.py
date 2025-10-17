import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GINConv, global_add_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=13, hidden=128, layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            mlp = nn.Sequential(nn.Linear(in_dim if i==0 else hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.layers.append(GINConv(mlp))
        self.out_dim = hidden

    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = torch.relu(conv(x, edge_index))
        return global_add_pool(x, batch)

class ChemTextEncoder(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, smiles_list):
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        return out.last_hidden_state[:,0,:]

class BioHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GNNEncoder()
        self.text = ChemTextEncoder()
        self.fc_fuse = nn.Sequential(nn.Linear(self.gnn.out_dim + self.text.out_dim, 256), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(nn.Linear(256*4,128), nn.ReLU(), nn.Linear(128,1), nn.Sigmoid())

    def forward_single(self, g, txt):
        g_emb = self.gnn(g.x, g.edge_index, g.batch)
        t_emb = self.text(txt)
        return self.fc_fuse(torch.cat([g_emb, t_emb], dim=1))

    def forward_pair(self, g1, txt1, g2, txt2):
        m1 = self.forward_single(g1, txt1)
        m2 = self.forward_single(g2, txt2)
        fused = torch.cat([m1,m2,m1*m2,torch.abs(m1-m2)], dim=1)
        return self.classifier(fused)
