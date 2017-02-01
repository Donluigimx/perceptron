from numpy import random


class Perceptron(object):

    def __init__(self, data, y=1, bias=0.5, x_range=10):
        self.data = [(value[0], value[y]) for value in data]
        self.weights = [random.random() for i in range(2)]
        self.threshold = 0.5
        self.bias = bias
        self.x_range = 10

    def training(self):
        while True:
            errors = 0
            for vector, d in self.data:
                result = Perceptron.vector_product(vector, self.weights, self.bias) > self.threshold
                error = d - result
                if error != 0:
                    errors += 1
                    for i, value in enumerate(vector):
                        if d == 1:
                            self.weights[i] += value
                        else:
                            self.weights[i] -= value
                    if d == 1:
                        self.bias += 1
                    else:
                        self.bias -= 1
            if errors == 0:
                break

    @staticmethod
    def vector_product(values, weights, bias):
        return sum(value * weight for value, weight in zip(values, weights)) + bias

    @property
    def y(self):
        return [(-self.bias-(self.weights[0]*x))/self.weights[1] for x in range(-self.x_range, self.x_range)]

    @property
    def x(self):
        return [x for x in range(-self.x_range, self.x_range)]
