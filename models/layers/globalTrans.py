import torch.nn as nn

class GlobalTransformer(nn.Module):
    def __init__(self, config):
        super(GlobalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

    def forward(self, x):
        # Transformer expects input in the format (L, N, E)
        x = x.permute(1, 0, 2)  # Permute to match the expected input format (sequence length, batch size, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Permute back to (batch size, sequence length, features)
        return x
