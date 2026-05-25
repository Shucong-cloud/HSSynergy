import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_修改 import (GAT_Block, CoAttentionLayer, drug_Interaction, Drug_Cell_In)
from torch_geometric.nn import LayerNorm
import pandas as pd

# SDDS model
'''num_features_xd=78, n_head=4, num_features_xt=954, output_dim=128, dropout=0.2, n_gats=6'''


class SDDSynergyNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_head=2, num_features_xt=954, output_dim=128, dropout=0.2, n_gats=6):
        super(SDDSynergyNet, self).__init__()

        # initial normal
        self.initial_norm = LayerNorm(num_features_xd)
        # graph drug convolution and drug layerNorm
        self.drug_gats = []
        self.drug_norm = []
        for i in range(n_gats):
            drug_gat = GAT_Block(n_head, num_features_xd, output_dim)
            self.add_module(f"drug_gat{i}", drug_gat)
            self.drug_gats.append(drug_gat)
            self.drug_norm.append(LayerNorm(output_dim * n_head).cuda())
            num_features_xd = output_dim * n_head
        self.drug_fc = nn.Linear(78, 256)

        # drug interaction
        self.co_attention = CoAttentionLayer(n_head * output_dim)
        self.interaction = drug_Interaction(n_head * output_dim, n_gats=n_gats, n_heads=n_head)

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * n_head),
            nn.ReLU()
        )

        # drug cell interaction - 双向交互机制
        self.drug_cell = Drug_Cell_In(output_dim * n_head)

        # combined layers
        self.fc1 = nn.Linear(n_gats ** 2 + 2 * n_gats, 128)
        self.fc2 = nn.Linear(128, 2)

        # activation and regularization
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.output_dim = output_dim
        self.n_gats = n_gats

    def forward(self, x1, x2, edge_index1, edge_index2, batch1, batch2, cell):
        # 处理药物1和药物2的分子图，通过多层GAT_Block提取子结构
        repr_drug1 = []
        repr_drug2 = []
        x1 = self.initial_norm(x1, batch1)
        x2 = self.initial_norm(x2, batch2)
        for i, drug_gat in enumerate(self.drug_gats):
            drug1 = drug_gat(x1, edge_index1, batch1)
            drug2 = drug_gat(x2, edge_index2, batch2)
            h_1 = drug1[0]  # x
            r_1 = drug1[1]  # emb
            h_2 = drug2[0]  # x
            r_2 = drug2[1]  # emb
            repr_drug1.append(r_1)
            repr_drug2.append(r_2)
            # 修改：LayerNorm不需要传入batch参数
            h1 = self.drug_norm[i](h_1)
            h2 = self.drug_norm[i](h_2)
            x1 = self.elu(h1)
            x2 = self.elu(h2)

        # 堆叠各层子结构嵌入，形成最终的子结构表示
        repr_drug1 = torch.stack(repr_drug1, dim=-2)  # [batch, n_gats, n_head*output_dim]
        repr_drug2 = torch.stack(repr_drug2, dim=-2)  # [batch, n_gats, n_head*output_dim]

        # 处理细胞特征
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        # 药物-细胞双向交互
        drug1_cell, tau1_bi = self.drug_cell(repr_drug1, cell_vector)
        drug2_cell, tau2_bi = self.drug_cell(repr_drug2, cell_vector)

        # 子结构筛选阶段：用双向交互分数优化子结构特征
        repr_drug1 = repr_drug1 * tau1_bi.unsqueeze(-1)  # [batch, n_gats, feature]
        repr_drug2 = repr_drug2 * tau2_bi.unsqueeze(-1)  # [batch, n_gats, feature]

        # 协同注意力和药物相互作用（子结构对交互阶段）
        co_attention = self.co_attention(repr_drug1, repr_drug2)
        drug_interaction = self.interaction(repr_drug1, repr_drug2, co_attention, tau1_bi, tau2_bi)

        # 拼接药物-细胞交互特征
        drug_cell = torch.cat((drug1_cell, drug2_cell), dim=1)

        # 拼接所有特征
        xc = torch.cat((drug_interaction, drug_cell), 1)
        xc = F.normalize(xc, 2, 1)

        # 全连接层
        xc = self.fc1(xc)
        xc = self.elu(xc)
        out = self.fc2(xc)
        return out