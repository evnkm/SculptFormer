import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.localTrans import LocalTransformer
from models.layers.globalTrans import GlobalTransformer  # Assuming these are defined

class P2MModel(nn.Module):
    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        # Initialize Encoder and Decoder from Backbone
        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        # Graph Convolutional Networks Initialization
        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        # Unpooling layers
        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        # Projection module
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        # Single Local Transformer
        self.local_transformer = LocalTransformer(options.local_transformer_config)

        # Graph Convolution Layer
        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Initial deformation process
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)
        x1_up = self.unpooling[0](x1)

        # Passing features through the single local transformer
        x = self.local_transformer(x1_up)
        x2, x_hidden = self.gcns[1](x)
        x2_up = self.unpooling[1](x2)

        # Final Graph Convolution Network
        x3, _ = self.gcns[2](x2_up)
        if self.gconv_activation:
            x3 = F.relu(x3)
        x3 = self.gconv(x3)

        # Optional Decoder for Reconstruction
        reconst = self.nn_decoder(img_feats) if self.nn_decoder is not None else None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst
        }

