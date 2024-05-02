from src import Neuron


class NeuronLink:
    def __init__(self, neuron: Neuron, weight: float):
        self.__next_neuron = neuron
        self.__weight = weight

    def get_weight(self):
        return self.__weight

    def get_next_neuron(self):
        return self.__next_neuron

    def set_weight(self, weight: float):
        self.__weight = weight

    def sum_weight(self, weight: float):
        self.__weight += weight