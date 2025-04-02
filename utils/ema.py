import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach().to(param.device)

    def update(self):
        """Update the shadow weights using Exponential Moving Average (EMA)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (self.decay * self.shadow[name].to(param.device) + (1.0 - self.decay) * param)

    def apply(self):
        """Apply the EMA weights to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].to(param.device)
