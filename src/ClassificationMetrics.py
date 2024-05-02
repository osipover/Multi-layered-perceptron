def accuracy(predictions, targets, positive_class, negative_class):
    check_lengths(predictions, targets)
    n = len(targets)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(n):
        if targets[i] == predictions[i] and predictions[i] == positive_class:
            true_positive += 1
        elif targets[i] != predictions[i] and predictions[i] == positive_class:
            false_positive += 1
        elif targets[i] == predictions[i] and predictions[i] == negative_class:
            true_negative += 1
        elif targets[i] != predictions[i] and predictions[i] == negative_class:
            false_negative += 1
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def precision(predictions, targets, positive_class):
    check_lengths(predictions, targets)
    n = len(targets)
    true_positive = 0
    false_positive = 0
    for i in range(n):
        if targets[i] == predictions[i] and predictions[i] == positive_class:
            true_positive += 1
        elif targets[i] != predictions[i] and predictions[i] == positive_class:
            false_positive += 1
    return true_positive / (true_positive + false_positive)


def recall(predictions, targets, positive_class, negative_class):
    check_lengths(predictions, targets)
    n = len(targets)
    true_positive = 0
    false_negative = 0
    for i in range(n):
        if targets[i] == predictions[i] and predictions[i] == positive_class:
            true_positive += 1
        elif targets[i] != predictions[i] and predictions[i] == negative_class:
            false_negative += 1
    return true_positive / (true_positive + false_negative)


def check_lengths(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")
