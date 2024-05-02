import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.Perceptron import Perceptron
from src.PerceptronUtils import normalize, un_normalize
import src.ClassificationMetrics as m
from math import fabs
from sklearn.metrics import auc, roc_curve

POISONOUS_CLASS = 112
EDIBLE_CLASS = 101

def create_train_and_test_sets(df: pd.DataFrame):
    total_num_rows = df.shape[0]
    train_num_rows = int(total_num_rows * 0.8)
    train_set = df[:train_num_rows]
    test_set = df[train_num_rows:]
    return train_set, test_set


def categorize_mushroom(value: float):
    if fabs(value - EDIBLE_CLASS) < fabs(value - POISONOUS_CLASS):
        return EDIBLE_CLASS
    else:
        return POISONOUS_CLASS


def educate_network(perceptron: Perceptron, train_set: pd.DataFrame):
    for i in range(100):
        perceptron.educate(train_set)


def test_network(perceptron: Perceptron, test_set: pd.DataFrame):
    predictions = []
    targets = []
    for i in range(len(test_set.values)):
        row = test_set.values[i]
        target = categorize_mushroom(un_normalize(row[-1], max_value, min_value))
        predict = categorize_mushroom(un_normalize(perceptron.predict(row), max_value, min_value))
        predictions.append(predict)
        targets.append(target)
    accuracy = m.accuracy(predictions, targets, POISONOUS_CLASS, EDIBLE_CLASS)
    precision = m.precision(predictions, targets, POISONOUS_CLASS)
    recall = m.recall(predictions, targets, POISONOUS_CLASS, EDIBLE_CLASS)

    label_encoder = LabelEncoder()
    y_true_binary = label_encoder.fit_transform(targets)
    fpr, tpr, thresholds = roc_curve(y_true_binary, predictions)
    auc_roc = auc(fpr, tpr)
    print('Accuracy = {}'.format(accuracy))
    print('Precision = {}'.format(precision))
    print('Recall = {}'.format(recall))
    print('AUC = {}'.format(auc_roc))

weights = [44.532705892292476, -69.12217970058428, 0.7007028083755391, 6.865692055182277, -35.75195886341987,
           -59.14777938584877, -1.9373935715582336, -5.734733970463104, -46.17956806132287, -39.72614061492847,
           1.159148777885414, 0.5399059494771123, 2.8074483726478836, -15.642587882071624, -7.668200297294016,
           -3.127754994613105, 13.930650701268837, -30.451142296464074, -4.433885960939668, -0.9391700059864742,
           0.13422555856729845, 13.213920622226276, -8.714818424774744, -10.405389314231861, 9.138707373456489,
           17.551268230356918, -5.175930052317558, -6.788560898456351, 22.35250683672736, 48.35205326790828,
           -4.63926616782786, -4.701649653178854, -33.42592626086079, 1.42823590695794, -9.720325983130644,
           -9.949604162645374, -46.38658048477958, -38.72389420829434, -11.839420974220682, -9.009469375604658,
           59.86679899725984, 2.306278542520665, -5.316582209262492, -5.9765688730850925, 15.717881637765993,
           64.30563529627638, -7.812792436213769, -9.843545463525176, -5.541381900020206, -13.38866650128513,
           -16.26666592640488, -21.77755133485246, 15.452729960108874, -14.186775696437842, -16.011295210637865,
           -31.057234322044227, 16.84462213275253, 7.482118097719047, 4.858802077154299, 3.97221225496046,
           3.8365673005222436, 4.453521084493105, 4.058555157255514, 4.878998660761103, -13.034907973543122,
           19.56997428871227, 15.769090403424594]

if __name__ == '__main__':
    data_set = pd.read_csv('./resources/mushrooms_filtered.csv')

    max_value = data_set['class'].max()
    min_value = data_set['class'].min()

    normalized_set = normalize(data_set)

    train_set, test_set = create_train_and_test_sets(normalized_set)

    perceptron = Perceptron.create_with_weights([len(train_set.columns) - 1, 4, 3], weights, 0.5)

    print('########## EDUCATION STARTED ##########')
    educate_network(perceptron, train_set)
    print('########## EDUCATION FINISHED ##########')

    print('############# TEST STARTED ############')
    test_network(perceptron, test_set)
    print('############# TEST FINISHED ############')
