import torch
import torch.nn.functional as F


class CauseEffect(torch.nn.Module):
    """ The main architercture, trained of predicting the
    causal effect of As on Y.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - outputSize (output dim)
     - hidden_dim (hidden dim)"""

    def __init__(self, args,
                 inputSize,
                 outputSize,
                 hidden_dim):
        super().__init__()
        self.args = args
        if args.dataset == 'PertImgSim':
            self.num_phis = args.Z_dim // (args.window_size_phi ** (2))
        else:
            self.num_phis = args.num_phis

        self.model = MultiTaskMLP(args,
                                  inputSize=inputSize,
                                  hidden_dim=hidden_dim,
                                  outputSize=outputSize,
                                  num_heads=self.num_phis)

    def forward(self, input):
        output = self.model(input)
        return output


class MultiTaskMLP(torch.nn.Module):
    """ Generic regression MLP model class.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)
     - num_heads (number of network heads)"""

    def __init__(self, args,
                 inputSize, hidden_dim,
                 outputSize, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden = torch.nn.Linear(inputSize, hidden_dim)
        self.relu = F.relu
        self.drop = torch.nn.Dropout(p=args.dropout)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = F.relu
        self.drop2 = torch.nn.Dropout(p=args.dropout)
        self.hidden3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = F.relu
        self.drop3 = torch.nn.Dropout(p=args.dropout)
        self.output = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = F.relu

        self.heads = []
        for i in range(self.num_heads):
            self.heads.append(torch.nn.Linear(hidden_dim, outputSize))
        self.heads = torch.nn.ModuleList(self.heads)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.hidden3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.output(x)
        x = self.relu4(x)
        outs = []
        for w in range(self.num_heads):
            outs.append(self.heads[w](x))
        return tuple(outs)


class MLP(torch.nn.Module):
    """ Generic regression MLP model class.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)"""

    def __init__(self, args, inputSize,
                 hidden_dim, outputSize):
        super().__init__()

        self.hidden = torch.nn.Linear(inputSize, hidden_dim)
        self.relu = F.relu
        self.drop = torch.nn.Dropout(p=args.dropout)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = F.relu
        self.drop2 = torch.nn.Dropout(p=args.dropout)
        self.hidden3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = F.relu
        self.drop3 = torch.nn.Dropout(p=args.dropout)
        self.output = torch.nn.Linear(hidden_dim, outputSize)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.hidden3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        out = self.output(x)
        return out
