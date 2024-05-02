import pandas as pd
import numpy as np


class GainRatio:
    def __init__(self, df: pd.DataFrame, target: str):
        self.__df = df
        self.__target = target

    def gain_ratio(self, attribute: pd.Series, target: pd.Series):
        return self.__gain(target, attribute) / self.__split_info(target, attribute)

    def gain_ratio_for_all(self):
        map = {}
        for attribute_name in self.__df:
            if attribute_name != self.__target:
                map[attribute_name] = self.gain_ratio(self.__df[attribute_name], self.__df[self.__target])
        return dict(sorted(map.items(), key=lambda item: item[1]))

    def __entropy(self, attribute: pd.Series):
        sum = 0
        values_count = attribute.value_counts()
        total = attribute.shape[0]
        for i in attribute.unique():
            p = values_count[i] / total
            sum += p * np.log2(p)
        return sum * (-1)


    def __entropy_info(self, target: pd.Series, attribute: pd.Series):
        sum = 0
        target_total = target.shape[0]
        counts = attribute.value_counts()
        df = pd.concat([target, attribute], axis=1)

        for i in attribute.unique():
            value_count = counts[i]
            for j in target.unique():
                details = df.apply(lambda x: True if x[attribute.name] == i and x[target.name] == j else False, axis=1)
                num = len(details[details == True].index)
                if num != 0:
                    sum -= (value_count / target_total) * (num / value_count) * np.log2(num / value_count)
        return sum


    def __convert_num_by_sturges(self, col_name: str, df: pd.DataFrame):
        col = df[col_name]
        bins = int(1 + np.log2(len(col.unique())))
        bin_width = col.max() / bins
        bins_labels = [i for i in range(bins + 1)]

        intervals = [0 + i * bin_width for i in range(bins + 1)]
        intervals.append(np.float64(np.inf))

        return pd.cut(df[col_name], intervals, labels=bins_labels)


    def __split_info(self, target: pd.Series, attribute: pd.Series):
        sum = 0
        total_target = target.shape[0]
        counts = attribute.value_counts()

        for value in attribute.unique():
            value_count = counts[value]
            if value_count != 0:
                sum -= (value_count / total_target) * np.log2(value_count / total_target)
        return sum


    def __gain(self, target: pd.Series, attribute: pd.Series):
        target_entropy = self.__entropy(target)
        attribute_entropy = self.__entropy_info(target, attribute)
        return target_entropy - attribute_entropy
