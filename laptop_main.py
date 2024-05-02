from math import sqrt

import pandas as pd

from src.Perceptron import Perceptron
from src.PerceptronUtils import un_normalize, normalize, create_train_and_test_sets
import src.RegressionMetrics as m


def educate_network(perceptron: Perceptron, train_set: pd.DataFrame):
    for i in range(500):
        perceptron.educate(train_set)


def test_network(perceptron: Perceptron, test_set: pd.DataFrame):
    predictions = []
    targets = []
    for i in range(len(test_set.values)):
        row = test_set.values[i]
        target = un_normalize(row[-1], max_value, min_value)
        predict = un_normalize(perceptron.predict(row), max_value, min_value)
        predictions.append(predict)
        targets.append(target)
    print('MAE: ' + str(m.mae(predictions, targets)))
    print('RMSE: ' + str(m.rmse(predictions, targets)))


weights = [-1.4720933576821928, -1.0159300958605793, -0.3882250839366267, -0.0865978952883887, -7.523738395933483,
           -2.5645359995824397, -1.4240072545321285, -0.7725437008191731, -15.773033902526187, -0.1806989157416492,
           -3.274118138185513, -6.6482346352862205, 30.404207882969107, 9.998976455549457, -4.21003118111387,
           16.498229459006534, -2.3432262078405537, -0.9353624188113089, -0.4797127138103424, -0.2175155827601992,
           -3.1237771782281, -0.6756387763192819, -0.6092866818624705, -0.8747865257451646, -3.22541220367984,
           -0.5862788890493763, -3.626515134328666, -2.004034002068752, 5.572613107758595, -2.186891281919542,
           26.035272076637074, -16.861285636120158, 20.1145693967109, -3.182828784074511, 5.305905437291628,
           -3.8020976603595713, -13.458410981446011, 19.86438850884385, -10.674494087950219]

if __name__ == '__main__':
    data_set = pd.read_csv('resources/Upd_Laptop_price.csv')

    max_value = data_set['Price'].max()
    min_value = data_set['Price'].min()

    normalized_set = normalize(data_set)

    train_set, test_set = create_train_and_test_sets(normalized_set)

    perceptron = Perceptron.create_with_weights([len(train_set.columns) - 1, 4, 3], weights, 0.5)

    print('########## EDUCATION STARTED ##########')
    educate_network(perceptron, train_set)
    print('########## EDUCATION FINISHED ##########')

    print('############# TEST STARTED ############')
    test_network(perceptron, test_set)
    print('############# TEST FINISHED ############')
