from isaaclab_rl.rsl_rl.exporter import _TorchPolicyExporter
import torch

class _CustomizedTorchPolicyExporter(_TorchPolicyExporter):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__(actor_critic, normalizer)
        self.actor = actor_critic.actor
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = actor_critic.memory_a.rnn
            self.rnn.cpu()

            # Check type of RNN
            if self.rnn.__class__.__name__ == "LSTM":
                self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory

            elif self.rnn.__class__.__name__ == "GRU":
                self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_gru
                self.reset = self.reset_memory_gru

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)
    
    def reset_memory_gru(self):
        self.hidden_state[:] = 0.0

    