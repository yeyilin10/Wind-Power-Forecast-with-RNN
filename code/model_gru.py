from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn

class BaselineGRUModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGRUModel, self).__init__()

        self.input_size = int(settings["in_var"])
        self.hidden_size = int(settings["gru_hidden_size"])
        
        self.in_features = int(settings["input_len"] / settings["step_size"] * self.hidden_size)
        self.out_features = int(settings["output_len"] * settings["out_var"])

        self.num_layers = int(settings["gru_layers"])
        self.dropout = settings["dropout"]
        self.gru1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, batch_first=True, dropout=self.dropout, 
                                        )

        self.projection = nn.Linear(in_features=self.in_features, out_features=self.out_features)


    def forward(self, x):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        x, _ = self.gru1(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.projection(x)

        return output # [Batch, *]


