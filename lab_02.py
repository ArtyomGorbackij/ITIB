from math import sqrt, pow
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self, real_values, window, teaching_rate):
        self.window = window
        self.points_num = 20
        self.w = [0] * (self.window + 1)
        self.real_val = real_values
        self.calculated = self.real_val[:self.window] + [0] * (self.points_num - self.window)
        self.nn = teaching_rate

    def train(self, max_iter):
        num = 0
        epsilon = 0
        while num < max_iter:
            for i in range(self.window, self.points_num):
                net = self.w[self.window]
                for j in range(self.window):
                    net += self.w[j] * self.real_val[i - self.window + j]
                self.calculated[i] = net

                for j in range(self.window):
                    self.w[j] += self.nn * (self.real_val[i] - self.calculated[i]) * self.real_val[i - self.window + j]

                epsilon = 0
                for j in range(self.window, self.points_num):
                    epsilon += pow(self.real_val[j] - self.calculated[j], 2)

                epsilon = sqrt(epsilon)

                if epsilon < 0.0001:
                    print(epsilon)
                    break

            num += 1
            print(num, epsilon, self.w)

    def predict(self, num):
        for i in range(self.points_num, self.points_num + num):
            net = self.w[self.window]
            for j in range(self.window):
                net += self.w[j] * self.calculated[i - self.window + j]

            self.calculated += [net]


def calculate(begin, end, points_quantity, f):
    step = (end - begin) / (points_quantity - 1)
    res = []
    points = []
    for i in range(points_quantity):
        points += [begin + step * i]
        res += [f(begin + step * i)]
    return points, res


def main():
    fun = lambda x: np.sin(x-1)
    a = -2
    b = 2
    x, real_values = calculate(a, b, 20, fun)
    x1, y1 = calculate(b, 2 * b - a, 20, fun)
    real_values += y1
    x += x1

    c = 400
    d = 4
    b = 0.5

    n = Neuron(real_values, d, b)
    n.train(c)

    n.predict(20)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.plot(x, real_values)
    plt.plot(x, n.calculated)
    plt.show()


main()
