from src import NeuronLink
from src.Neuron import Neuron


class InputNeuron(Neuron):
    def __init__(self, links: [NeuronLink]):
        super().__init__(links)

    def get_activation_data(self):
        return self._sum
