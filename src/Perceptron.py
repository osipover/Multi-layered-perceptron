import random

import pandas as pd
from numpy import ndarray

from src.Neuron import Neuron
from src.ActivationNeuron import ActivationNeuron
from src.NeuronLink import NeuronLink
from src.InputNeuron import InputNeuron


class Perceptron:
    MIN_WEIGHT = -0.5
    MAX_WEIGHT = 0.5

    def __init__(self, layers: [[Neuron]], education_speed):
        self.__layers = layers
        self.__education_speed = education_speed

    @staticmethod
    def create_with_weights(num_neurons_in_layers: [int], weights: [float], education_speed):
        index_w = 0
        output_neuron = ActivationNeuron(None)
        neurons = [[output_neuron]]
        cur_layer = [output_neuron]
        for index, count in enumerate(reversed(num_neurons_in_layers)):
            next_layer = cur_layer.copy()
            cur_layer = []
            for i in range(count):
                links_to_next_layer = []
                for n in next_layer:
                    links_to_next_layer.append(NeuronLink(n, weights[index_w]))
                    index_w += 1
                neuron = InputNeuron(links_to_next_layer) if index == len(num_neurons_in_layers) - 1 \
                    else ActivationNeuron(links_to_next_layer)
                cur_layer.append(neuron)
            neurons.append(cur_layer)
        return Perceptron(neurons[::-1], education_speed)

    @staticmethod
    def create_default(num_neurons_in_layers: [int], education_speed):
        output_neuron = ActivationNeuron(None)
        neurons = [[output_neuron]]
        cur_layer = [output_neuron]
        for index, count in enumerate(reversed(num_neurons_in_layers)):
            next_layer = cur_layer.copy()
            cur_layer = []
            for i in range(count):
                links_to_next_layer = []
                for n in next_layer:
                    weight = random.uniform(Perceptron.MIN_WEIGHT, Perceptron.MAX_WEIGHT)
                    links_to_next_layer.append(NeuronLink(n, weight))
                neuron = InputNeuron(links_to_next_layer) if index == len(num_neurons_in_layers) - 1 \
                    else ActivationNeuron(links_to_next_layer)
                cur_layer.append(neuron)
            neurons.append(cur_layer)
        return Perceptron(neurons[::-1], education_speed)

    def predict(self, row):
        self.__fill(row)
        predict = self.__run()
        self.__clean()
        return predict

    def get_layers(self) -> [[Neuron]]:
        return self.__layers

    def educate(self, data_set: pd.DataFrame):
        for i in range(len(data_set.values)):
            row = data_set.values[i]
            self.__fill(row)
            self.__run()
            self.__local_gradient(row[-1])
            self.__study()
            self.__clean()

    def __fill(self, row: ndarray):
        for i, input_neuron in enumerate(self.__layers[0]):
            input_neuron.set_data(row[i])

    def __run(self):
        for layer in self.__layers[:-1]:
            for neuron in layer:
                neuron.run()
        return self.__get_output()

    def __local_gradient(self, target_value):
        output_neuron = self.__layers[-1][-1]
        output_local_grad = self.__calc_output_local_grad(output_neuron.get_activation_data(), target_value)
        output_neuron.set_error(output_local_grad)
        for layer in reversed(self.__layers[1:-1]):
            for neuron in layer:
                next_local_grad = self.__calc_next_local_grad(neuron)
                local_grad = self.__calc_local_grad(neuron.get_activation_data(), next_local_grad)
                neuron.set_error(local_grad)

    def __calc_local_grad(self, activation_data: float, next_error: float):
        return next_error * activation_data * (1 - activation_data)

    def __calc_next_local_grad(self, neuron: Neuron):
        links = neuron.get_links()
        error = 0
        for link in links:
            next_neuron = link.get_next_neuron()
            error += next_neuron.get_error() * link.get_weight()
        return error

    def __get_output(self):
        return self.__layers[-1][-1].get_activation_data()

    def __calc_output_local_grad(self, output_value: float, target_value: float):
        return (output_value - target_value) * output_value * (1 - output_value)

    def __study(self):
        for layer in reversed(self.__layers[:-1]):
            for neuron in layer:
                self.__update_weight(neuron)

    def __update_weight(self, neuron: Neuron):
        for link in neuron.get_links():
            next_neuron = link.get_next_neuron()
            error = next_neuron.get_error()
            activation_data = neuron.get_activation_data()
            dif = -self.__education_speed * error * activation_data
            link.sum_weight(dif)

    def __clean(self):
        for layer in self.__layers[1:]:
            for neuron in layer:
                neuron.set_data(0)
