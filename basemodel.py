import torch.nn as nn
from abc import abstractmethod
from fvcore.nn import FlopCountAnalysis
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def param_num(self, str):
        return sum([param.nelement() for param in self.parameters()])

    def flops(self, inputs):
        flops = FlopCountAnalysis(self, inputs)
        return flops.total()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
