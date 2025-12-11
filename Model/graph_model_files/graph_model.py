import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import logging
import numpy as np

__all__ = ["GraphConfig", "EmbeddingGNN"]

logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    # Data & Temporal Configuration
    input_dims: Tuple[int, ...] = field(default_factory=lambda: (64,))
    mode: str = "3m"
    window_size: int = 9
    dim_opt: int = 3

    # Temporal Encoder (TCN + GRU / ATTENTION)
    hidden_size: int = 128              # GRU/Attention internal dim
    num_layers: int = 1                 # GRU/Attention layers
    dropout: float = 0.1
    tconv_channels: int = 128           # TCN output channels
    use_temporal_attention: bool = False #
    
    # Spatial Encoder (GNN)
    gnn_hidden: int = 128
    gnn_layers: int = 2
    
    # Graph Topology Configuration 
    graph_type: str = "diffusion"
    use_adaptive_graph: bool = False
    adaptive_dim: int = 10
    
    # Diffusion Specific
    K: int = 2
    dropedge_p: float = 0.05
    
    feature_groups: List[int] = field(default_factory=lambda: [1])

    # Graph Tensors
    n_nodes: Optional[int] = None
    A_dense: Optional[torch.Tensor] = None

class _FFDimPicker:
    """Helper to pick hardcoded projection dimensions."""
    @staticmethod
    def pick(num_inputs: int, dim_opt: int) -> List[int]:
        if num_inputs == 4:
            table = {1:[48,48,64,4], 2:[48,48,128,4], 3:[48,64,128,4], 4:[64,48,64,4]}
            return table.get(dim_opt, table[1])
        if num_inputs == 5:
            table = {1:[48,48,48,48,4], 2:[48,48,64,64,4], 3:[48,48,64,64,4], 4:[64,48,48,48,4]}
            return table.get(dim_opt, table[1])
        if num_inputs == 6:
            table = {1:[48,48,48,48,48,4], 2:[48,48,128,48,48,4], 3:[48,48,128,48,48,4], 4:[64,48,128,48,48,4]}
            return table.get(dim_opt, table[1])
        if num_inputs == 3: return [48,128,4]
        if num_inputs == 2: return [128,4]
        return [64]*(num_inputs-1)+[4]

def _add_self_loops_and_norm(A: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize adjacency matrix."""
    N = A.size(0)
    device = A.device
    I = torch.eye(N, device=device, dtype=A.dtype)
    A = A + I
    deg = A.sum(dim=1).clamp(min=1e-12)
    inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def _row_normalize(A: torch.Tensor) -> torch.Tensor:
    """Row-normalize to get random-walk matrix P."""
    deg = A.sum(dim=1).clamp(min=1e-12)
    return A / deg.unsqueeze(1)

class DropEdge(nn.Module):
    """Applies DropEdge to a dense adjacency."""
    def __init__(self, p: float):
        super().__init__()
        self.p = float(max(0.0, min(0.99, p)))
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return A
        keep = (torch.rand_like(A) > self.p).to(A.dtype)
        return (A * keep) / (1.0 - self.p)

# ---------- Temporal Blocks (TCN and new Attention) ----------
class TemporalConvBlock(nn.Module):
    """TCN-like temporal block (used before GRU/Attention)."""
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.dw1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dw2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B,W,N,C = X.shape
        Y = X.reshape(B*N, W, C).transpose(1,2)      # [B*N, C, W]
        R = Y
        Y = self.pw1(self.act(self.dw1(Y)))
        Y = self.drop(Y)
        Y = self.pw2(self.act(self.dw2(Y))) + R
        Y = Y.transpose(1,2).reshape(B, W, N, C)
        Y = self.ln(Y + X)
        return Y

class TemporalAttentionBlock(nn.Module):
    """
    NEW SOTA Module: Temporal Self-Attention Encoder (replaces GRU).
    Processes sequence [W, C] and outputs the last time step's feature vector.
    """
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # Use a standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4, # A fixed number of heads is often sufficient
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.proj = nn.Linear(input_dim, hidden_size) if input_dim != hidden_size else nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X is (B*N, W, C_in)
        H_sequence = self.transformer_encoder(X) # H_sequence is (B*N, W, C_in)
        
        H0_flat = H_sequence[:, -1, :] # H0_flat is (B*N, C_in)
        
        return self.proj(H0_flat) # H0_flat_projected is (B*N, H)

# ---------- Graph Block (Diffusion) ----------
class DiffusionGraphBlock(nn.Module):
    """Diffusion graph conv: sum_{k=0..K} (Theta_k^f P^k H + Theta_k^b (P^T)^k H)."""
    def __init__(self, in_dim: int, out_dim: int, K: int, dropout: float, dropedge_p: float):
        super().__init__()
        self.K = K
        self.theta_f = nn.ParameterList([nn.Parameter(torch.Tensor(in_dim, out_dim)) for _ in range(K+1)])
        self.theta_b = nn.ParameterList([nn.Parameter(torch.Tensor(in_dim, out_dim)) for _ in range(K+1)])
        self.bias    = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropedge = DropEdge(dropedge_p)

    def reset_parameters(self):
        for w in list(self.theta_f) + list(self.theta_b):
            nn.init.xavier_uniform_(w)

    def forward(self, H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        B, N, Cin = H.shape
        Pf = self.dropedge(P)
        Pb = self.dropedge(P.transpose(0,1))
        
        acc = 0
        Hk_f = H
        Hk_b = H
        for k in range(self.K+1):
            acc_f = torch.matmul(Hk_f, self.theta_f[k])
            acc_b = torch.matmul(Hk_b, self.theta_b[k])
            acc = acc + acc_f + acc_b
            if k < self.K:
                Hk_f = torch.matmul(Pf, Hk_f)
                Hk_b = torch.matmul(Pb, Hk_b)

        Z = acc + self.bias
        Z = self.act(Z)
        Z = self.drop(Z)
        return self.norm(Z + self.res_proj(H))


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import logging
import numpy as np
from torch.utils.data import Dataset # Added to resolve DataLoader import issues

__all__ = ["GraphConfig", "EmbeddingGNN"]

logger = logging.getLogger(__name__)



@dataclass
class GraphConfig:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    tconv_channels: int = 128
    use_temporal_attention: bool = False
    
    feature_groups: List[int] = field(default_factory=lambda: [1])

    n_nodes: Optional[int] = None
    A_dense: Optional[torch.Tensor] = None


class EmbeddingGNN(nn.Module):
    """Spatio-Temporal GNN with Hierarchical Feature Fusion."""
    def __init__(self, config: GraphConfig, output_dim: int = 1):
        super().__init__()
        assert config.A_dense is not None or config.use_adaptive_graph, "GraphConfig requires adjacency."
        assert config.n_nodes is not None, "GraphConfig requires n_nodes."

        self.config = config
        self.N = config.n_nodes
        self.prediction_months = {"1m":1, "3m":3, "6m":6}[config.mode]
        self.output_dim = output_dim

        # --- 1. Per-embedding projection ---
        num_inputs = len(config.input_dims)
        embed_dims = _FFDimPicker.pick(num_inputs, config.dim_opt)
        self.embedding_networks = nn.ModuleList()
        self.embedding_bns = nn.ModuleList()

        for in_dim, out_dim in zip(config.input_dims, embed_dims):
            net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(config.dropout))
            self.embedding_networks.append(net)
            self.embedding_bns.append(nn.LayerNorm(out_dim))
        
        # --- Group feature dimensions ---
        group_dims = {}
        for i, group_id in enumerate(config.feature_groups):
            group_dims[group_id] = group_dims.get(group_id, 0) + embed_dims[i]

        # --- 2. Temporal Encoders (Multiple per group) ---
        self.temporal_encoders = nn.ModuleDict()
        self.tproj_groups = nn.ModuleDict()
        self.tblock_groups = nn.ModuleDict()
        
        # Build one TCN + GRU/Attention stack for each unique feature group
        for group_id, total_dim in group_dims.items():
            t_in = config.tconv_channels
            
            # Projection for the concatenated group features
            self.tproj_groups[str(group_id)] = nn.Linear(total_dim, config.tconv_channels)
            self.tblock_groups[str(group_id)] = TemporalConvBlock(config.tconv_channels, config.dropout)
            
            if config.use_temporal_attention:
                encoder = TemporalAttentionBlock(t_in, config.hidden_size, config.num_layers, config.dropout)
            else:
                encoder = nn.GRU(t_in, config.hidden_size, config.num_layers, 
                                 dropout=config.dropout if config.num_layers > 1 else 0.0, batch_first=True)
            
            self.temporal_encoders[str(group_id)] = encoder
        
        # GNN input size is the sum of hidden sizes from all encoders
        gnn_in_dim = len(group_dims) * config.hidden_size
        
        # --- 3. Graph Construction (Static or Adaptive) ---
        if config.use_adaptive_graph:
            self.E1 = nn.Parameter(torch.randn(self.N, config.adaptive_dim))
            self.E2 = nn.Parameter(torch.randn(config.adaptive_dim, self.N))
        else:
            A_sym = _add_self_loops_and_norm(config.A_dense.to(torch.float32))
            self.register_buffer("P_rw",  _row_normalize(A_sym))

        # --- 4. Graph Encoder ---
        gblocks = []
        last = gnn_in_dim # Input is the concatenated result of all temporal encoders
        if config.graph_type == 'diffusion':
            for _ in range(config.gnn_layers):
                gblocks.append(DiffusionGraphBlock(last, config.gnn_hidden, config.K, config.dropout, config.dropedge_p))
                last = config.gnn_hidden
        self.gnn = nn.ModuleList(gblocks)

        # --- 5. Multi-task heads ---
        self.fc_outs = nn.ModuleList([nn.Linear(last, self.prediction_months) for _ in range(self.output_dim)])

    def _project_embeddings(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Projects each embedding and returns a list of projected features."""
        B, W, N, _ = embeddings[0].shape
        projected_feats = []
        for emb, net, ln in zip(embeddings, self.embedding_networks, self.embedding_bns):
            x = emb.reshape(-1, emb.size(-1))
            x = net(x)
            x = x.reshape(B, W, N, -1)
            x = ln(x)
            projected_feats.append(x)
        return projected_feats

    def _get_P_matrix(self) -> torch.Tensor:
        """Calculates the P matrix (row-stochastic) for graph propagation."""
        # ... (Adaptive/Static logic is assumed unchanged) ...
        if self.config.use_adaptive_graph:
            A_adaptive = F.relu(torch.mm(self.E1, self.E2))
            if self.config.A_dense is not None:
                A_static = self.config.A_dense.to(A_adaptive.device, dtype=A_adaptive.dtype)
                A_combined = A_adaptive + A_static.detach()
            else:
                A_combined = A_adaptive
            A_sym = _add_self_loops_and_norm(A_combined)
            P_rw = _row_normalize(A_sym)
            return P_rw
        else:
            return self.P_rw

    # --- forward ---
    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        B, W, N, _ = embeddings[0].shape
        
        # 1) Project ALL embeddings
        projected_feats = self._project_embeddings(list(embeddings))
        
        # 2) Group and run Temporal Encoders (Hierarchical Fusion)
        group_h0s = []
        unique_groups = sorted(list(set(self.config.feature_groups)))
        
        for group_id in unique_groups:
            group_id_str = str(group_id)
            # a) Identify features belonging to this group and concatenate them
            group_feats = []
            for i, feat in enumerate(projected_feats):
                if self.config.feature_groups[i] == group_id:
                    group_feats.append(feat)
            
            X_group = torch.cat(group_feats, dim=-1) # (B, W, N, C_group)
            
            # b) TCN pre-processing
            X_group = self.tproj_groups[group_id_str](X_group)
            X_group = self.tblock_groups[group_id_str](X_group)
            
            # c) Temporal Encoding (GRU or Attention)
            Xr_group = X_group.permute(0, 2, 1, 3).contiguous().view(B * N, W, -1) # (B*N, W, C_in)
            
            temporal_encoder = self.temporal_encoders[group_id_str]
            
            if self.config.use_temporal_attention:
                H0_flat = temporal_encoder(Xr_group) # Attention returns (B*N, H)
            else:
                _, h = temporal_encoder(Xr_group)
                H0_flat = h[-1] # GRU returns (L, B*N, H), take last layer
            
            group_h0s.append(H0_flat)
        
        # Concatenate all group hidden states (H0) for Spatial Mixing
        H0_final_flat = torch.cat(group_h0s, dim=-1) # (B*N, H_total)
        H0 = H0_final_flat.view(B, N, -1)            # (B, N, H_total)
        
        # 3) Graph propagation
        P = self._get_P_matrix()
        H = H0
        for layer in self.gnn:
            H = layer(H, P)

        # 4) Multi-task, multi-horizon heads
        Hf = H.view(B * N, H.size(-1))
        outs = []
        for head in self.fc_outs:
            y = head(Hf).view(B, N, self.prediction_months)
            outs.append(y)
        Y = torch.stack(outs, dim=-1).permute(0, 2, 1, 3).contiguous()
        return Y