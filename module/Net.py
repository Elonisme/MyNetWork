from module.Function import *
from module.Layer import Layer


class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)

    def forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:
                Layer_output = ReLU(layer_sum)
                Layer_output = normalize(Layer_output)
            else:
                Layer_output = Softmax(layer_sum)
            outputs.append(Layer_output)
        return outputs