import numpy as np


class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.
        self.mse_sum = 0.
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum
        self.mse_sum += n_sum * n_sum
        self.n_counts += n_counts

    def add_tpfpfn(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def add_loss(self, l1, l2, l3, l4, l5, n_counts):
        self.n_sum += l1
        self.mse_sum += l2
        self.tp += l3
        self.fp += l4
        self.fn += l5
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts

    def get_mse(self):
        return self.mse_sum / self.n_counts

    def get_f1(self):
        recall, precision, f1 = 1., 1., 1.
        if self.tp + self.fn != 0:
            recall = self.tp / (self.tp + self.fn)
        if self.tp + self.fp != 0:
            precision = self.tp / (self.tp + self.fp)
        if precision + recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def get_avg_loss(self):
        n = self.n_counts
        return self.n_sum / n, self.mse_sum / n, self.tp / n, self.fp / n, self.fn / n
