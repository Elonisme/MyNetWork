import numpy as np
from module.Function import classfify
from module.Net import Network
from tools.generated_data import create_data, plot_data


if __name__ == "__main__":
    NETWORK_SHAPE = [2, 3, 4, 2]
    NUM_OF_DATA = 100

    # inputs
    data = create_data(NUM_OF_DATA)
    plot_data(data, "Right classification")

    inputs = data[:,(0,1)]
    print(inputs)

    network = Network(NETWORK_SHAPE)
    outputs = network.forward(inputs)
    print(outputs[-1])
    classification = classfify(outputs[-1])
    print(classification)
    data[:,2] = classification
    print(data)

    plot_data(data, "Before training")