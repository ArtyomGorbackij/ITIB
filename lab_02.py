from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


def fun(x):
    return np.sin(x-1)


def net(w, values, i):
    return w[p] + sum(w[j] * values[i - p + j] for j in range(p))


def learn(epoches_count) -> np.array:
    w = np.zeros(p + 1)
    epoch = 0
    eps = 1000
    while epoch < epoches_count:
        for i in range(p, n):
            y_[i] = net(w, y, i)
            d = (y[i] - y_[i])
            for j in range(p):
                w[j] += norm * d * y[i - p + j]
            eps = sum((y[j] - y_[j]) ** 2 for j in range(p, n))
            if sqrt(eps) < 0.0001:
                break
        epoch += 1
    return w, eps


def predict(w, num):
    predicted_values = y_
    for i in range(n, n + num):
        net_ = net(w, predicted_values, i)
        predicted_values = np.append(predicted_values, [net_])
    return predicted_values


if __name__ == "__main__":
    a = -2
    b = 2
    p = 6
    norm = 0.5
    n = 20
    x = np.append(np.linspace(a, b, 20), np.linspace(b, 2 * b - a, 20))
    y = fun(x)
    y_ = np.append(y[:p], np.zeros(n - p))
    # Обучение на М=4000 эпох
    (w_, eps_) = learn(4000)
    predicted_values_ = predict(w_, 20)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.grid()
    plt.plot(x, y)
    plt.plot(x[20:], predicted_values_[20:], "o")
    plt.show()
    # Исследование зависимости погрешности от числа эпох
    e = list()
    for i in range(1, 300, 1):
        y_ = np.append(y[:p], np.zeros(n - p))
        (w_, eps_) = learn(i)
        e.append(eps_)
    plt.xlabel("Количество эпох")
    plt.ylabel("Погрешность")
    plt.grid()
    plt.plot(range(1, 300, 1), e)
    plt.show()
    print(e[len(e) - 1])
    # Исследование зависимости погрешности от нормы обучения
    e = list()
    for i in np.arange(0.3, 1, 0.05):
        norm = i
        y_ = np.append(y[:p], np.zeros(n - p))
        (w_, eps_) = learn(4000)
        e.append(eps_)
    norm = 0.5
    plt.xlabel("Норма обучения")
    plt.ylabel("Погрешность")
    plt.grid()
    plt.plot(np.arange(0.3, 1, 0.05), e)
    plt.show()
    # Исследование зависимости погрешности от длины окна
    e = list()
    for i in range(4, 16):
        p = i
        y_ = np.append(y[:p], np.zeros(n - p))
        (w_, eps_) = learn(4000)
        e.append(eps_)
    p = 6
    plt.xlabel("Длина окна")
    plt.ylabel("Погрешность")
    plt.grid()
    plt.plot(range(4, 16), e)
    plt.show()
