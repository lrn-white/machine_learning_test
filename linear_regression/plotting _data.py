from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def matplob_test(x_data, y_data):
    plt.scatter(x_data, y_data, color="blue")
    x_data = [[i] for i in x_data]
    y_data = [[i] for i in y_data]
    model = linear_model.LinearRegression()
    model.fit(x_data, y_data)

    y2 = model.predict(x_data)
    plt.plot(x_data, y2, color='green')

    # plt.axis([2.5, 20, -5, 20])
    # plt.show()


def personal_test(x_data, y_data, ep=0.0001, max_iter=10000):
    m = len(x_data)
    alpha = 0.001

    k = np.random.random()
    b = np.random.random()

    for i in range(50000):
        grandK = (sum([(k * x_data[i] + b - y_data[i]) * x_data[i] for i in range(m)])) / m
        grandB = (sum([(k * x_data[i] + b - y_data[i]) for i in range(m)])) / m

        tempK = k - alpha * grandK
        tempB = b - alpha * grandB
        k = tempK
        b = tempB

    y_data_predict = [[i * k + b] for i in x_data]

    plt.plot(x_data, y_data_predict, color='red')

    plt.axis([2.5, 20, -5, 20])
    plt.show()


def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = len(x)  # 数据的行数

    # 初始化参数(theta)
    t0 = np.random.random()
    t1 = np.random.random()

    # 代价函数, J(theta)
    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    # 进行迭代
    while not converged:
        # 计算训练集中每一行数据的梯度 (d/d_theta j(theta))
        grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        # 更新参数 theta
        # 此处注意，参数要同时进行更新，所以要建立临时变量来传值
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
        t0 = temp0
        t1 = temp1

        # 均方误差 (MSE)
        e = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True

        J = e  # 更新误差值
        iter += 1  # 更新迭代次数

        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    y_data_predict = [i * t1 + t0 for i in x]

    plt.plot(x, y_data_predict, color='red')

    plt.axis([2.5, 20, -5, 20])
    plt.show()


if __name__ == '__main__':
    # 假设函数：y= kx+b
    mpl.use("TkAgg")

    with open("machine-learning-ex1/ex1/ex1data1.txt", "r") as f:
        data = f.readlines()

    x_data = [float(i.split(",")[0]) for i in data]
    y_data = [float(i.split(",")[1].replace("\n", '')) for i in data]

    matplob_test(x_data, y_data)

    personal_test(x_data, y_data)
    # gradient_descent(0.001, x_data, y_data)
