import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.helpers import SinusoidalPosEmb, init_weights

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def functional_forward(self, state, action, parameters):
        """Stateless twin-Q forward used by the shared compiled graph."""
        x = torch.cat([state, action], dim=-1)
        outputs = []
        offset = 0
        for _ in range(2):
            for layer in range(4):
                x = F.linear(x, parameters[offset], parameters[offset + 1])
                offset += 2
                if layer != 3:
                    x = F.mish(x)
            outputs.append(x)
            x = torch.cat([state, action], dim=-1)
        return outputs[0], outputs[1]

class CriticQR(nn.Module):
    """TQC-style ensemble of distributional quantile critics.

    Every layer stores all ensemble weights in one parameter. This preserves
    independent critics while allowing Inductor to lower each layer to one
    batched GEMM, including its backward pass.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 n_quantiles=25, n_nets=5):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        dimensions = (
            (state_dim + action_dim, hidden_dim),
            (hidden_dim, hidden_dim),
            (hidden_dim, hidden_dim),
            (hidden_dim, n_quantiles),
        )
        packed_layers = []
        for input_dim, output_dim in dimensions:
            linears = [nn.Linear(input_dim, output_dim) for _ in range(n_nets)]
            for linear in linears:
                init_weights(linear)
            packed_layers.extend((
                nn.Parameter(torch.stack([linear.weight for linear in linears])),
                nn.Parameter(torch.stack([linear.bias for linear in linears])),
            ))
        self.layers = nn.ParameterList(packed_layers)

    def forward(self, state, action):
        return self.functional_forward(
            state, action, tuple(self.layers)
        )

    def q(self, state, action):
        return self.forward(state, action).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

    def functional_forward(self, state, action, parameters):
        """Stateless packed forward shared by online and target critics."""
        x = torch.cat([state, action], dim=-1)
        x = x.unsqueeze(0).expand(self.n_nets, -1, -1)
        for layer in range(4):
            weight, bias = parameters[2 * layer:2 * layer + 2]
            x = torch.bmm(x, weight.transpose(1, 2)) + bias[:, None, :]
            if layer != 3:
                x = F.mish(x)
        return x.permute(1, 0, 2)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load packed checkpoints or migrate the previous ``nets.*`` layout."""
        if any(key.startswith('nets.') for key in state_dict):
            converted = {}
            for layer, module_index in enumerate((0, 2, 4, 6)):
                converted[f'layers.{2 * layer}'] = torch.stack([
                    state_dict[f'nets.{net}.{module_index}.weight']
                    for net in range(self.n_nets)
                ])
                converted[f'layers.{2 * layer + 1}'] = torch.stack([
                    state_dict[f'nets.{net}.{module_index}.bias']
                    for net in range(self.n_nets)
                ])
            state_dict = converted
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, time_dim=32):
        super(Model, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = state_dim + action_dim + time_dim
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, action_dim))
        self.apply(init_weights)
        

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)

        return out

    def functional_forward(self, x, time, state, parameters):
        """Stateless denoiser forward shared by actor and target actor graphs."""
        frequencies = self.time_mlp[0].frequencies
        t = time[:, None] * frequencies[None, :]
        t = torch.cat((t.sin(), t.cos()), dim=-1)
        t = F.linear(t, parameters[0], parameters[1])
        t = F.mish(t)
        t = F.linear(t, parameters[2], parameters[3])

        out = torch.cat([x, t, state], dim=-1)
        offset = 4
        for layer in range(4):
            out = F.linear(out, parameters[offset], parameters[offset + 1])
            offset += 2
            if layer != 3:
                out = F.mish(out)
        return out

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(MLP, self).__init__()

        input_dim = state_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish())
        
        self.final_layer = nn.Linear(hidden_size, action_dim)

        self.apply(init_weights)

    def forward(self, state, eval=False, q_func=None):
        out = self.mid_layer(state)
        out = self.final_layer(out)

        if not eval:
            out += torch.randn_like(out) * 0.1

        return out

    def loss(self, action, state, weights=1.0):
        return weighted_mse_loss(self.forward(state), action, weights)

    def sample_n(self, state, times=32, chosen=1, q_func=None):
        raise NotImplemented
