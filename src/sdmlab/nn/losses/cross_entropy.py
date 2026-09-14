
from .base import Loss      # or just Layer directly — your call whether a separate base is worth it
from ...autograd.ops import CrossEntropy as CrossEntropyFunction

class CrossEntropy(Loss):
    def forward(self, logits, target):
        return CrossEntropyFunction.apply(logits, target)
