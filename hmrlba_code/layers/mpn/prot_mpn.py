import torch
import torch.nn as nn
from typing import Dict
from torch_geometric.nn import global_add_pool, global_mean_pool

from hmrlba_code.layers import mpn_layer_from_config
from hmrlba_code.utils.tensor import build_mlp, index_select_ND


class ProtMPN(nn.Module):

    def __init__(self,
                 mpn_config: Dict,
                 encoder: str,
                 prot_mode: str = 'backbone',
                 use_attn: bool = False,
                 graph_pool: str = 'sum_pool',
                 use_mpn_in_patch: bool = False,
                 **kwargs):
        super(ProtMPN, self).__init__(**kwargs)
        self.encoder = encoder
        self.mpn_config = mpn_config
        self.prot_mode = prot_mode
        self.use_attn = use_attn
        self.graph_pool = graph_pool
        self.use_mpn_in_patch = use_mpn_in_patch
        self._build_components()

    def _build_components(self):

        if self.prot_mode == 'surface2backbone':
            bconfig = self.mpn_config['bconfig']
            sconfig = self.mpn_config['sconfig']

            self.backbone_mpn = mpn_layer_from_config(bconfig, self.encoder)
            self.surface_mpn = mpn_layer_from_config(sconfig, self.encoder)


        else:
            raise ValueError(f"{self.prot_mode} is currently not supported.")

    def run_component_mpn(self, data):
        component = getattr(data, self.prot_mode)
        inputs = (component.x, component.edge_index, component.edge_attr)

        node_emb = self.mpn(*inputs)
        return node_emb

    def run_mpn_in_patch(self, data):
        node_emb = self.surface_to_patch_mpn(data.x, data.edge_index, data.edge_attr)
        patch_emb = global_mean_pool(node_emb, data.patch_members)
        return patch_emb

    def run_full_mpn(self, data):
        order = self.prot_mode.split("2")
        bottom_layer, top_layer = order

        bottom_graph = getattr(data, bottom_layer)
        top_graph = getattr(data, top_layer)
        hier_mapping = data.mapping

        x = bottom_graph.x
        # This is a bit hacky for now since we are just prototyping

        bottom_mpn_inputs = (x, bottom_graph.edge_index, bottom_graph.edge_attr)
        if self.encoder in ['gru', 'lstm']:
            bottom_mpn_inputs += (bottom_graph.mess_idx,)

        bottom_layer_mpn = getattr(self, bottom_layer + "_mpn")
        hnode_bottom = bottom_layer_mpn(*bottom_mpn_inputs)

        x = top_graph.x

        top_mpn_inputs = (x, top_graph.edge_index, top_graph.edge_attr)

        top_layer_mpn = getattr(self, top_layer + "_mpn")
        hnode_top = top_layer_mpn(*top_mpn_inputs)

        return hnode_top, hnode_bottom

    def forward(self, data):
        if self.prot_mode in ['backbone', 'surface']:
            node_emb = self.run_component_mpn(data)
        else:
            top_node_emb, bottom_node_emb = self.run_full_mpn(data)

        top_component = self.prot_mode.split("2")[-1]  # Aggregate node embeddings of last layer
        bottom_component = self.prot_mode.split("2")[0]  # Aggregate node embeddings of last layer
        if self.graph_pool == "sum_pool":
            top_graph_emb = global_add_pool(top_node_emb, getattr(data, top_component).batch)
            bottom_graph_emb = global_add_pool(bottom_node_emb, getattr(data, bottom_component).batch)

        elif self.graph_pool == "mean_pool":
            top_graph_emb = global_mean_pool(top_node_emb, getattr(data, top_component).batch)
            bottom_graph_emb = global_mean_pool(bottom_node_emb, getattr(data, bottom_component).batch)

        return top_graph_emb, bottom_graph_emb
