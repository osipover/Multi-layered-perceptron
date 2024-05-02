import math

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import Perceptron


def sigmoid(x: float, alpha: float) -> float:
    return 1 / (1 + math.exp(-alpha * x))


def normalize(ds: pd.DataFrame):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(ds)
    return pd.DataFrame(norm_data, columns=ds.columns)


def un_normalize(val: float, max_value: float, min_value: float) -> float:
    return (max_value - min_value) * val + min_value


def create_train_and_test_sets(df: pd.DataFrame):
    total_num_rows = df.shape[0]
    train_num_rows = int(total_num_rows * 0.8)
    train_set = df[:train_num_rows]
    test_set = df[train_num_rows:]
    return train_set, test_set


def output_weights(perceptron: Perceptron):
    layers = perceptron.get_layers()
    weights = []
    for layer in layers:
        for neuron in layer:
            links = neuron.get_links()
            if links is not None:
                for link in links:
                    weights.append(link.get_weight())
    print(weights)