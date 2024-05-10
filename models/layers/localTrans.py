import torch.nn as nn

class LocalTransformer(nn.Module):
    def __init__(self, config, local_window_size=5):
        super(LocalTransformer, self).__init__()
        self.local_window_size = local_window_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # Create a local mask that only allows attending within a window around each position
        seq_len, batch_size, _ = x.size()
        full_mask = self.generate_square_subsequent_mask(seq_len)
        window_mask = full_mask.clone()
        
        # Ensure each position can only see its local neighbors within the specified window size
        for i in range(seq_len):
            window_mask[i, max(0, i - self.local_window_size): min(seq_len, i + self.local_window_size + 1)] = 0
        
        x = x.permute(1, 0, 2)  # Permute to (sequence length, batch size, features)
        x = self.transformer_encoder(x, mask=window_mask)
        x = x.permute(1, 0, 2)  # Permute back to (batch size, sequence length, features)
        return x
