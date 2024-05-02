from math import sqrt


def rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")
    rmse = 0
    n = len(targets)
    for i in range(n):
        rmse += (targets[i] - predictions[i]) ** 2
    return sqrt((1 / n) * rmse)


def mae(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("targets and predictions must be the same length")
    mae = 0
    n = len(targets)
    for i in range(n):
        mae += abs(targets[i] - predictions[i])
    return mae / n

