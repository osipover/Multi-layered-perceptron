
import abc

from src import NeuronLink


class Neuron:
    def __init__(self, links: [NeuronLink]):
        self.__links = links
        self._sum: float = 0
        self.__error: float = 0

    def run(self):
        value: float = self.get_activation_data()
        for link in self.get_links():
            neuron = link.get_next_neuron()
            neuron.add_data(value * link.get_weight())

    @abc.abstractmethod
    def get_activation_data(self):
        pass

    def add_data(self, value: float):
        self._sum += value

    def set_data(self, value: float):
        self._sum = value

    def get_data(self):
        return self._sum

    def get_links(self):
        return self.__links

    def get_error(self):
        return self.__error

    def set_error(self, error: float):
        self.__error = error
