import torch
import torch.nn as nn

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes: int, key='class'):
        super().__init__()
        self.key = key
        self.n_classes = n_classes
        # we add one more class for the unconditionnal class
        self.embedding = nn.Embedding(n_classes+1, embed_dim)

    def forward(self, batch):
        c = batch[:, None].int()
        return self.embedding(c)

    def get_unconditional_class(self, batch_size):
        """
        Consider a classifier with N classes, ranging from 0 to N-1.
        We add one more class, which corresponds to the unconditional class.
        Thus, class N corresponds to the unconditional case.
        """
        return torch.full((batch_size,), self.n_classes)