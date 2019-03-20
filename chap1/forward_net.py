import numpy as np
import forward_layer

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # Initialize
        W1, b1 = np.random.randn(I, H), np.random.randn(H)
        W2, b2 = np.random.randn(H, O), np.random.randn(O)

        # Make layers
        self.layers = [
            forward_layer.Affine(W1, b1),
            forward_layer.Sigmoid(),
            forward_layer.Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    # forward = predict
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
