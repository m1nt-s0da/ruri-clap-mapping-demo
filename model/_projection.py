from torch import nn

__all__ = ["RuriClapProjectionModel"]


class RuriClapProjectionModel(nn.Module):
    def __init__(
        self,
        dim_ruri: int = 256,
        dim_clap: int = 512,
        dim_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim_ruri, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_clap),
        )

    def forward(self, x):
        return self.seq(x)
