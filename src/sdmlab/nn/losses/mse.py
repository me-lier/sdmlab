
from .base import Loss      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import MSE as MSEFunction

class MSE(Loss):
    def forward(self, pred, target):
        return MSEFunction.apply(pred, target)
