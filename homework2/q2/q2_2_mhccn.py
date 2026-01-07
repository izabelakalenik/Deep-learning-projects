import torch.nn as nn

class RNA_CNN_MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=4):
        super().__init__()


        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        

        # ---- Multi-head self-attention ----
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(64)
        self.global_pool = nn.AdaptiveMaxPool1d(1)


        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, x):
        #               B   L  C
        # x shape: (Batch, 41, 4)
        x = x.permute(0, 2, 1)  # (Batch, 4, 41)

        x = self.pool(self.relu(self.conv1(x)))  
        x = self.relu(self.conv2(x))              

        x = x.permute(0, 2, 1)  # (Batch, Length, 64)

        # Self-attention (Q = K = V)
        attn_out, _ = self.attention(x, x, x, need_weights=False)

        # Residual connection + normalization
        x = self.attn_norm(x + attn_out)

        x = x.permute(0, 2, 1)            # (Batch, 64, Length)
        x = self.global_pool(x).squeeze(-1)  # (Batch, 64)

        return self.fc(x)