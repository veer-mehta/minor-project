"""Building blocks of the UPT interaction head.

Attribute names follow the authors' released checkpoints so their weights load directly.
"""
import torch
import torch.nn.functional as F
from torch import nn


def pair_spatial_features(b1, b2, height, width, eps=1e-10):
    """Hand-crafted layout features for box pairs. b1, b2: (M, 4) xyxy. Returns (M, 36)."""
    c1_x, c1_y = (b1[:, 0] + b1[:, 2]) / 2, (b1[:, 1] + b1[:, 3]) / 2
    c2_x, c2_y = (b2[:, 0] + b2[:, 2]) / 2, (b2[:, 1] + b2[:, 3]) / 2
    w1, h1 = b1[:, 2] - b1[:, 0], b1[:, 3] - b1[:, 1]
    w2, h2 = b2[:, 2] - b2[:, 0], b2[:, 3] - b2[:, 1]

    inter = (torch.min(b1[:, 2:], b2[:, 2:]) - torch.max(b1[:, :2], b2[:, :2])).clamp(min=0).prod(1)
    iou = inter / (w1 * h1 + w2 * h2 - inter)

    # Offset of the second box from the first, in units of the first box's size
    d_x = (c2_x - c1_x).abs() / (w1 + eps)
    d_y = (c2_y - c1_y).abs() / (h1 + eps)

    f = torch.stack([
        c1_x / width, c1_y / height, c2_x / width, c2_y / height,  # centres
        w1 / width, h1 / height, w2 / width, h2 / height,  # sizes
        w1 * h1 / (height * width), w2 * h2 / (height * width),  # areas
        w2 * h2 / (w1 * h1 + eps),  # area ratio
        w1 / (h1 + eps), w2 / (h2 + eps),  # aspect ratios
        iou,
        (c2_x > c1_x) * d_x, (c2_x < c1_x) * d_x,  # right of / left of
        (c2_y > c1_y) * d_y, (c2_y < c1_y) * d_y,  # below / above
    ], 1)
    return torch.cat([f, torch.log(f + eps)], 1)


class FeedForward(nn.Module):
    def __init__(self, size, inner_size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(size, inner_size), nn.GELU(), nn.Linear(inner_size, size))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(size, eps=1e-12)

    def forward(self, x):
        return self.norm(x + self.dropout(self.ffn(x)))


class Fusion(nn.Module):
    """Multi-branch fusion: combine appearance and spatial features through parallel small branches."""

    def __init__(self, appearance_size, spatial_size, out_size, branches=16):
        super().__init__()
        inner = out_size // branches
        self.fc_1 = nn.ModuleList(nn.Linear(appearance_size, inner) for _ in range(branches))
        self.fc_2 = nn.ModuleList(nn.Linear(spatial_size, inner) for _ in range(branches))
        self.fc_3 = nn.ModuleList(nn.Linear(inner, out_size) for _ in range(branches))

    def forward(self, appearance, spatial):
        return F.relu(sum(
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ))


class CooperativeLayer(nn.Module):
    """Attention over box tokens where every pair's weight also depends on the pair's spatial encoding."""

    def __init__(self, size=256, inner_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head = inner_size // num_heads
        self.unary = nn.Linear(size, inner_size)
        self.pairwise = nn.Linear(inner_size, inner_size)
        self.attn = nn.ModuleList(nn.Linear(3 * head, 1) for _ in range(num_heads))
        self.message = nn.ModuleList(nn.Linear(head, head) for _ in range(num_heads))
        self.aggregate = nn.Linear(inner_size, size)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(size, size * 4, dropout)

    def forward(self, x, y):
        """x: (N, size) box tokens. y: (N, N, inner_size) pairwise spatial encodings."""
        n = len(x)
        u = F.relu(self.unary(x)).view(n, self.num_heads, -1)  # (N, H, L)
        p = F.relu(self.pairwise(y)).view(n, n, self.num_heads, -1)  # (N, N, H, L)
        sender = u[:, None].expand(n, n, -1, -1)  # [i, j] -> token i
        receiver = u[None].expand(n, n, -1, -1)  # [i, j] -> token j
        features = torch.cat([sender, receiver, p], -1)

        heads = []
        for h in range(self.num_heads):
            weights = self.attn[h](features[:, :, h]).softmax(0)  # normalised over senders
            messages = self.message[h](sender[:, :, h] * p[:, :, h])
            heads.append((weights * messages).sum(0))
        update = self.aggregate(F.relu(torch.cat(heads, -1)))
        return self.ffn(self.norm(x + self.dropout(update)))


class CompetitiveLayer(nn.Module):
    """Standard transformer encoder layer over the pair tokens of one image."""

    def __init__(self, size=1024, inner_size=2048, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.ModuleDict({
            "attention": nn.ModuleDict({k: nn.Linear(size, size) for k in ("query", "key", "value")}),
            "output": nn.ModuleDict({"aggregate": nn.Linear(size, size), "norm": nn.LayerNorm(size, eps=1e-12)}),
        })
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(size, inner_size, dropout)

    def forward(self, x):
        """x: (M, size) pair tokens."""
        proj, out = self.attention["attention"], self.attention["output"]
        q, k, v = (proj[name](x).view(len(x), self.num_heads, -1).transpose(0, 1) for name in ("query", "key", "value"))
        m = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        m = m.transpose(0, 1).reshape(len(x), -1)
        return self.ffn(out["norm"](x + self.dropout(out["aggregate"](m))))
