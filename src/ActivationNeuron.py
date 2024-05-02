from src.NeuronLink import NeuronLink
from src.Neuron import Neuron
from src.PerceptronUtils import sigmoid


class ActivationNeuron(Neuron):
    def __init__(self, links: [NeuronLink]):
        super().__init__(links)

    def get_activation_data(self):
        return self.__activation_function(self._sum)

    def __activation_function(self, x: float):
        return sigmoid(x, 0.25)
