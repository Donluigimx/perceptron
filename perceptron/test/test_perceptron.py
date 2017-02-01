from perceptron import Perceptron
import matplotlib
matplotlib.rcParams['backend'] = "Qt5Agg"
from matplotlib import pyplot as plt

def test_multiclass():

    training_data = [
        ((0.5, 1.4), 0, 0),
        ((0.7, 1.8), 0, 0),
        ((0.8, 1.6), 0, 0),
        ((1.5, 0.8), 0, 1),
        ((2.0, 1.0), 0, 1),
        ((0.3, 0.5), 1, 0),
        ((0.0, 0.2), 1, 0),
        ((-0.3, 0.8), 1, 0),
        ((-0.5, -1.5), 1, 1),
        ((-1.5, -2.2), 1, 1),
    ]

    plot_values = [
        ['ro', 'bo'],
        ['rs', 'bs'],
    ]

    p1 = Perceptron(data=training_data)
    p2 = Perceptron(data=training_data, y=2)
    p1.training()
    p2.training()

    for data in training_data:
        print(data)
        plt.plot([data[0][0]], [data[0][1]], plot_values[data[1]][data[2]])
    for data in training_data:
        assert (Perceptron.vector_product(data[0], p1.weights, p1.bias) > p1.threshold) == data[1]
        assert (Perceptron.vector_product(data[0], p2.weights, p2.bias) > p1.threshold) == data[2]
    plt.plot(p1.x, p1.y)
    plt.plot(p2.x, p2.y)
    plt.axis([-5, 6, -5, 6])
    plt.show()

if __name__ == '__main__':
    test_multiclass()