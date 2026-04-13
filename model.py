from torch import nn
from torch.nn.functional import normalize
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # x: [B, D] for feature vectors
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class SEModule1D(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.Linear(dim, inner_dim),
            inner_act(),
            nn.Linear(inner_dim, dim),
            out_act(),
        )
        
    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale1D(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return self.weight * x + self.bias


class OverviewGlobalFusion(nn.Module):
    """
    第一阶段：Overview - 全局多视图感知
    使用大感受野（相当于大卷积核）进行全局视图融合
    """
    def __init__(self, dim, num_views, global_kernel_size=13):
        super().__init__()
        
        self.dim = dim
        self.num_views = num_views
        self.global_kernel_size = global_kernel_size
        
        # 全局视图感知投影
        self.global_query = nn.Sequential(
            nn.Linear(dim * num_views, dim),  # 处理所有视图的拼接特征
            nn.LayerNorm(dim),
        )
        
        self.global_key = nn.Sequential(
            nn.Linear(dim * num_views, dim),
            nn.LayerNorm(dim),
        )
        
        self.global_value = nn.Sequential(
            nn.Linear(dim * num_views, dim),
            nn.LayerNorm(dim),
        )
        
        # 模拟大卷积核的全局感受野
        self.global_weight_proj = nn.Linear(dim, global_kernel_size**2)
        
        # 全局特征提取
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            GRN(dim),
        )
        
        # 全局上下文编码
        self.global_context = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
    def forward(self, view_features):
        """
        Args:
            view_features: List of [B, D] tensors
        Returns:
            global_context: [B, D] - 全局上下文表示
        """
        # 拼接所有视图
        concat_views = torch.cat(view_features, dim=1)  # [B, V*D]
        
        # 全局Query-Key-Value计算
        global_q = self.global_query(concat_views)  # [B, D]
        global_k = self.global_key(concat_views)    # [B, D]
        global_v = self.global_value(concat_views)  # [B, D]
        
        # 计算全局注意力权重（模拟大卷积核）
        attention_logits = torch.sum(global_q * global_k, dim=-1, keepdim=True)  # [B, 1]
        global_weights = self.global_weight_proj(global_q)  # [B, kernel_size^2]
        global_weights = torch.softmax(global_weights, dim=-1)  # [B, kernel_size^2]
        
        # 全局特征聚合（模拟大卷积核的效果）
        weighted_value = global_v.unsqueeze(-1) * global_weights.unsqueeze(1)  # [B, D, kernel_size^2]
        global_features = torch.sum(weighted_value, dim=-1)  # [B, D]
        
        # 全局特征提取和增强
        global_features = self.global_feature_extractor(global_features)
        global_context = self.global_context(global_features)
        
        return global_context


class LocalRefinementFusion(nn.Module):
    """
    第二阶段：Look-Closely - 基于全局上下文的局部细化
    使用小感受野（相当于小卷积核）进行局部细化
    """
    def __init__(self, dim, num_views, local_kernel_size=5):
        super().__init__()
        
        self.dim = dim
        self.num_views = num_views
        self.local_kernel_size = local_kernel_size
        
        # 视图特异性局部处理
        self.view_local_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            ) for _ in range(num_views)
        ])
        
        # 全局上下文指导的局部权重生成
        self.context_guided_weights = nn.Sequential(
            nn.Linear(dim, dim * num_views),  # 全局上下文指导
            nn.LayerNorm(dim * num_views),
            nn.Tanh(),  # 生成指导信号
        )
        
        # 局部注意力计算
        self.local_query = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
        )
        
        self.local_key = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
        )
        
        # 局部权重投影（小卷积核）
        self.local_weight_proj = nn.Linear(dim // 2, local_kernel_size**2)
        
        # 局部特征融合
        self.local_fusion = nn.Sequential(
            nn.Linear(dim * num_views, dim),
            GRN(dim),
        )
        
        # SE增强
        self.se_module = SEModule1D(dim)
        
        # 最终输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
    def forward(self, view_features, global_context):
        """
        Args:
            view_features: List of [B, D] tensors
            global_context: [B, D] - 来自Overview阶段的全局上下文
        Returns:
            refined_features: [B, D] - 细化后的融合特征
        """
        B = view_features[0].shape[0]
        
        # 全局上下文指导权重生成
        guidance_weights = self.context_guided_weights(global_context)  # [B, V*D]
        guidance_weights = guidance_weights.view(B, self.num_views, self.dim)  # [B, V, D]
        
        # 对每个视图进行局部处理
        local_processed_views = []
        for i, view_feat in enumerate(view_features):
            # 应用视图特异性投影
            local_feat = self.view_local_projections[i](view_feat)  # [B, D]
            
            # 应用全局上下文指导
            guided_feat = local_feat * guidance_weights[:, i, :]  # [B, D]
            local_processed_views.append(guided_feat)
        
        # 局部注意力计算
        concat_local = torch.cat(local_processed_views, dim=1)  # [B, V*D]
        
        # 计算局部Query和Key
        local_q = self.local_query(global_context)  # [B, D//2] - 使用全局上下文作为query
        local_k = self.local_key(global_context)    # [B, D//2]
        
        # 局部权重计算（模拟小卷积核）
        local_attention = torch.sum(local_q * local_k, dim=-1, keepdim=True)  # [B, 1]
        local_weights = self.local_weight_proj(local_q)  # [B, local_kernel_size^2]
        local_weights = torch.softmax(local_weights, dim=-1)  # [B, local_kernel_size^2]
        
        # 局部特征融合
        refined_features = self.local_fusion(concat_local)  # [B, D]
        
        # 应用局部权重（模拟小卷积核的局部细化效果）
        weighted_refined = refined_features.unsqueeze(-1) * local_weights.unsqueeze(1)  # [B, D, local_kernel_size^2]
        refined_features = torch.sum(weighted_refined, dim=-1)  # [B, D]
        
        # SE增强
        refined_features = self.se_module(refined_features)
        
        # 与全局上下文融合
        refined_features = refined_features + global_context
        
        # 最终输出
        refined_features = self.output_proj(refined_features)
        
        return refined_features


class OverLoCKMultiViewFusion(nn.Module):
    """
    OverLoCK-inspired Multi-View Fusion Module
    实现真正的"先整体后局部"仿生学架构
    """
    def __init__(self, dim, num_views, global_kernel=13, local_kernel=5):
        super().__init__()
        
        self.dim = dim
        self.num_views = num_views
        
        # 视图预处理
        self.view_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_views)])
        
        # 第一阶段：Overview（全局感知）
        self.overview_fusion = OverviewGlobalFusion(
            dim=dim, 
            num_views=num_views, 
            global_kernel_size=global_kernel
        )
        
        # 第二阶段：Look-Closely（局部细化）
        self.local_refinement = LocalRefinementFusion(
            dim=dim, 
            num_views=num_views, 
            local_kernel_size=local_kernel
        )
        
        # 层级缩放
        self.global_scale = LayerScale1D(dim, init_value=1.0)
        self.local_scale = LayerScale1D(dim, init_value=0.1)
        
    def forward(self, view_features):
        """
        Args:
            view_features: List of [B, D] tensors, one for each view
        Returns:
            fused_features: [B, D] - 融合后的特征
        """
        # 预处理：归一化各视图
        normalized_views = []
        for i, view_feat in enumerate(view_features):
            normalized_views.append(self.view_norms[i](view_feat))
        
        # 第一阶段：Overview - 全局多视图感知
        global_context = self.overview_fusion(normalized_views)
        global_context = self.global_scale(global_context)
        
        # 第二阶段：Look-Closely - 基于全局上下文的局部细化
        refined_features = self.local_refinement(normalized_views, global_context)
        refined_features = self.local_scale(refined_features)
        
        # 最终融合：全局 + 局部
        fused_features = global_context + refined_features
        
        return fused_features


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)



class GCFAggMVC(nn.Module):
    """
    Multi-View Clustering with ContMix-inspired fusion mechanism
    """
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(GCFAggMVC, self).__init__()
        
        # Encoders and Decoders for each view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        
        # View-specific projections
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        
        # Common view projection
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        
        self.view = view
        
        

        # 替换self-attention为OverLoCK-inspired融合模块
        self.overlock_fusion = OverLoCKMultiViewFusion(
            dim=low_feature_dim,
            num_views=view,
            global_kernel=13,  # 大感受野用于全局感知
            local_kernel=5     # 小感受野用于局部细化
        )
        
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return xrs, zs, hs
    
    def GCFAgg(self, xs):
        """
        Graph Convolution Free Aggregation with ContMix fusion
        """
        zs = []
        Alist = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            A = self.computeA(F.normalize(x), mode='knn')
            Alist.append(A)
        
        # Apply ContMix-inspired fusion instead of self-attention
        commonz = self.overlock_fusion(zs)
        commonz = normalize(self.Common_view(commonz), dim=1)
        
        # Return attention weights for compatibility (set to None since we don't use attention)
        S = torch.mean(torch.stack(Alist), dim=0)
        return commonz, S
    

    def computeA(self, x, mode):
        if mode == 'cos':
            a = F.normalize(x, p=2, dim=1)
            b = F.normalize(x.T, p=2, dim=0)
            A = torch.mm(a, b)
            A = (A + 1) / 2
        if mode == 'kernel':
            x = torch.nn.functional.normalize(x, p=1.0, dim=1)
            a = x.unsqueeze(1)
            A = torch.exp(-torch.sum(((a - x.unsqueeze(0)) ** 2) * 1000, dim=2))
        if mode == 'knn':
            dis2 = (-2 * x.mm(x.t())) + torch.sum(torch.square(x), axis=1, keepdim=True) + torch.sum(
                torch.square(x.t()), axis=0, keepdim=True)
            A = torch.zeros(dis2.shape).cuda()
            A[(torch.arange(len(dis2)).unsqueeze(1), torch.topk(dis2, 10, largest=False).indices)] = 1
            A = A.detach()
        if mode == 'sigmod':
            A = 1/(1+torch.exp(-torch.mm(x, x.T)))
        return A

# Usage example and test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example parameters
    num_views = 3
    input_sizes = [100, 150, 200]  # Different input dimensions for each view
    low_feature_dim = 128
    high_feature_dim = 64
    batch_size = 32
    
    # Create model
    model = GCFAggMVC(
        view=num_views,
        input_size=input_sizes,
        low_feature_dim=low_feature_dim,
        high_feature_dim=high_feature_dim,
        device=device
    ).to(device)
    
    # Create dummy data
    xs = []
    for i in range(num_views):
        xs.append(torch.randn(batch_size, input_sizes[i]).to(device))
    
    # Test forward pass
    xrs, zs, hs = model(xs)
    print("Reconstruction shapes:", [xr.shape for xr in xrs])
    print("Encoded features shapes:", [z.shape for z in zs])
    print("Specific view features shapes:", [h.shape for h in hs])
    
    # Test fusion
    commonz, S = model.GCFAgg(xs)
    print("Common features shape:", commonz.shape)
    print("Fusion completed successfully!")
