class Model:
    def __init__(self):
        self.dense = dense(512, 50, weight_initializer= glorot_normal)

    def __call__(self, x):
        return self.dense(x)

    @property
    def parameters(self):
        return self.dense.parameters
