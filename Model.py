import torch.nn as nn
#class Model:
#    def __init__(self):
#        self.dense = dense(512, 50, weight_initializer= glorot_normal)

#    def __call__(self, x):
#        return self.dense(x)

#    @property
#    def parameters(self):
#        return self.dense.parameters

class Model(nn.Module):
    def __init__(self):
        # it is important to call `super` here to initialize your pytorch-model
        super(Model, self).__init__()
        self.dense = nn.Linear(512, 50)

    def forward(self, x):
        """
        Performs a forward pass, sending a batch of data through the model.

        Parameters
        ----------
        x : pytorch.Tensor
            The batch of input data

        Returns
        -------
        pytorch.Tensor
            The results of the model's forward pass"""
        # instead of __call__, we need to define `forward` for pytorch
        return self.dense(x)

    # there is no need to define a `parameters` property. PyTorch is pretty slick - its
    # `nn.Module` class will automatically "see" when you have set a neural net layer
    # as an attribute. `model.parameters()` will automatically return all of the trainable
    # parameters in your model
