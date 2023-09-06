import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as sp


def read_var(name, N, col=None, LR=None):
    if LR == 3:
        if col == 'Y':
            return list(pd.read_excel(name, sheet_name='Часть 3')[f'Unnamed: {2*N}'])
        return list(pd.read_excel(name, sheet_name='Часть 3')[f'Вариант {N}'])
    elif LR == 2:
        return list(pd.read_excel(name, sheet_name='Часть 1 и 2')[f'Вариант {N}'])


def print_line(leng):
    print('-' * leng)


# print dataframe
def print_df(df):
    c = df.shape[0] + 1
    print_line(len(df.to_string()) // c)
    print(f'{df.to_string()}')
    print_line(len(df.to_string()) // c)


def get_h(s):
    n = len(s)
    items = set(s)

    h = [s.count(i) for i in items]
    h_otn = [i / n for i in h]

    return h, h_otn


def get_var(s):
    return pd.DataFrame(data=s, columns=['Значения']).T


def get_stat(s):
    items = sorted(list(set(s)))
    h, h_otn = get_h(s)

    return pd.DataFrame(data=[items, h, h_otn], index=['Значения', 'Частоты', 'Относительная частота'])


def task1(s):
    var = get_var(s)
    stat = get_stat(s)

    print('Вариационный ряд:')
    print_df(var)
    print('Статистический ряд:')
    print_df(stat)
    print(f'Размах выборки:\t{s[-1] - s[0]}')


def get_mid_points(otr):
    count = len(otr)
    a, b = otr[0][0], otr[-1][1]
    step = otr[0][1] - otr[0][0]

    return [a + i * step - step / 2 for i in range(1, count + 1)]


def get_j(s, otr):
    s = sorted(s)
    count = len(otr)

    # Нахождение частот
    i, k = 0, 0
    j = [0 for i in range(count)]

    if not len(s):
        return j

    j[0] = 1 if s[0] == otr[0][0] else 0

    while k != count and i != len(s):
        x = s[i]
        if otr[k][0] < x <= otr[k][1]:
            j[k] += 1
            i += 1
        elif x > otr[k][1]:
            k += 1
        else:
            i += 1

    return j


def get_interval_stat(s, count, a=None, b=None):
    s = sorted(s)
    if not a and not b:
        a, b = s[0], s[-1]
    n = len(s)

    # Разбиение выборки на отрезки
    buf = np.linspace(a, b, count + 1)
    otr = [(buf[i - 1], buf[i]) for i in range(1, len(buf))]

    j = get_j(s, otr)

    # Нахождение относительных частот
    h_interval = [c / n for c in j]

    return otr, j, h_interval


def get_pd_interval_stat(interval_stat):
    return pd.DataFrame(data=interval_stat, index=['Отрезок', 'Частота', 'Отн. частота'])


def plot_interval_stat(s, mid_points):
    count = len(mid_points)

    step = mid_points[1] - mid_points[0]
    points = [mid_points[i//2] + step/2 if i % 2 else mid_points[i//2] - step/2 for i in range(2*count)]

    h, h_otn = get_h(s)
    otr, j, h_interval = get_interval_stat(s, count)

    value = [sum(h_interval[0:(i//2 + 1)]) for i in range(2*count)]

    plt.subplot(1, 3, 1)
    res = plt.hist(s, bins=count)
    plt.plot(mid_points, res[0])
    plt.grid()
    plt.title('Гистограмма частот X')
    plt.xlabel('Значения')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 2)
    res = plt.hist(s, bins=count, weights=h_otn)
    plt.plot(mid_points, res[0])
    plt.grid()
    plt.title('Гистограмма относительных частот X')
    plt.xlabel('Значения')
    plt.ylabel('Относительная частота')

    plt.subplot(1, 3, 3)
    plt.plot(points, value)
    plt.grid()
    plt.title('Эмпирическая функция Fx(X)')
    plt.xlabel('Значения')
    plt.ylabel('Fx(x)')

    plt.show()


def task2(s, mid_points, interval_stat):
    # Вывод таблицы
    interval_stat = pd.DataFrame(data=interval_stat, index=['Отрезок', 'Частота', 'Отн. частота'])
    print('Интервальный статистический ряд')
    print_df(interval_stat)

    # построение гистограммы частот
    plot_interval_stat(s, mid_points)


def get_mean(s: list) -> float:
    n = len(s)

    return float(sum(s) / n)


def get_group_mean(mid_points, j, n):
    count = len(mid_points)

    return sum([mid_points[i]*j[i] for i in range(count)])/n


def get_D(s: list, mean: float) -> float:
    n = len(s)

    return float(sum([(s[i] - mean) ** 2 for i in range(n)])/n)


def get_D_gruop(mid_points, j, n, mean_group):
    count = len(mid_points)

    return sum([(mid_points[i] - mean_group) ** 2 * j[i] for i in range(count)]) / n


def get_S_squared(n, D):
    return n * D / (n - 1)


def get_moda(s, h):
    moda = []
    n = len(s)

    for i in range(n):
        if h[i] == max(h):
            moda += [s[i]]

    return moda


def get_moda_group(mid_points, j):
    count = len(j)
    step = mid_points[1] - mid_points[0]

    max_indexes = []
    for i in range(count):
        if j[i] == max(j):
            max_indexes += [i]

    a_d = sum([mid_points[i] for i in max_indexes]) / len(max_indexes) - step / 2
    n_d = sum([j[i] for i in max_indexes]) / len(max_indexes)
    n_d_1 = j[max_indexes[0] - 1]
    n_d_2 = j[max_indexes[-1] + 1]
    return a_d + ((n_d - n_d_1) / (2 * n_d - n_d_1 - n_d_2)) * step


def get_median(s):
    n = len(s)
    i = n // 2

    return s[i + 1] if n % 2 else 1 / 2 * (s[i] + s[i + 1])


def get_median_group(n, mid_points, j):
    count = len(j)
    step = mid_points[1] - mid_points[0]

    i = count // 2
    x_l = mid_points[i] if count % 2 else 1 / 2 * (mid_points[i + 1] + mid_points[i])

    return (x_l - step / 2) + ((n / 2 - sum([j[k] for k in range(i)])) / j[i]) * step


def get_k(*args):
    n = len(args[0])
    k = n

    for s in args:
        k *= get_mean(s)

    return k


def get_k_group(X, count, N, A=None, B=None):
    n = len(count)
    k = N[0]

    for i in range(n):
        otr, j, h = get_interval_stat(X[i], count[i], a=A[i], b=B[i])
        mid = get_mid_points(otr)

        k *= get_group_mean(mid, j, N[i])

    return k


def get_cov(*args):
    def get_i(index):
        res = 1

        for s in args:
            res *= s[index]

        return res

    n = len(args[0])
    k = get_k(*args)

    return 1/(n - 1) * (sum([get_i(i) for i in range(n)]) - k)


def get_cov_group(h, X, X_mid, N, A, B):
    count_x = len(X_mid[0])
    count_y = len(X_mid[1])
    X, Y = X[0], X[1]
    res = 0

    for i in range(count_x):
        for j in range(count_y):
            res += h[j, i]*X_mid[0][i]*X_mid[1][j]

    return 1/(len(X) - 1) * (res - get_k_group([X, Y], (count_x, count_y), N, A=A, B=B))


def get_pxy(X):
    S = 1

    for x in X:
        S *= get_S_squared(len(x), get_D(x, get_mean(x)))

    return get_cov(*X) / S ** (1 / 2)


def get_pxy_gruop(X, count, h, A, B):
    S = 1
    n = len(X)

    otr, j, h_interval = get_interval_stat(X[0], a=A[0], b=B[0], count=count[0])
    mid_points_x = get_mid_points(otr)

    otr, j, h_interval = get_interval_stat(X[1], a=A[1], b=B[1], count=count[1])
    mid_points_y = get_mid_points(otr)

    for i in range(n):
        x = X[i]
        otr, j, h_interval = get_interval_stat(x, count=count[i], a=A[i], b=B[i])
        mid_points = get_mid_points(otr)

        S *= get_S_squared(len(x), get_D_gruop(mid_points, j, len(x), get_group_mean(mid_points, j, len(x))))

    return get_cov_group(h, (X[0], X[1]), (mid_points_x, mid_points_y), (len(X[0]), len(X[0])), A, B) / S**(1/2)


def get_params(X, count, A, B, h, cov, p, mean, moda, variance, unbiased_variance, median):
    def get_params(s, count, data, a, b, index=0):
        n = len(s)

        otr, j, h_interval = get_interval_stat(s, a=a, b=b, count=count)
        h, h_otn = get_h(s)
        mid_points = get_mid_points(otr)

        if mean:
            m = get_mean(s)
            m_group = get_group_mean(mid_points, j, n)
            Mx = (m, m_group)

            data[f'M{index}'] = Mx

        if variance:
            m = get_mean(s)
            m_group = get_group_mean(mid_points, j, n)

            D = get_D(s, m)
            D_group = get_D_gruop(mid_points, j, n, m_group)
            Dx = (D, D_group)

            data[f'D{index}'] = Dx

        if unbiased_variance:
            m = get_mean(s)
            m_group = get_group_mean(mid_points, j, n)

            D = get_D(s, m)
            D_group = get_D_gruop(mid_points, j, n, m_group)

            S = get_S_squared(n, D)
            S_group = get_S_squared(n, D_group)
            Sx = (S, S_group)

            data[f'S{index}'] = Sx

        if median:
            hx = get_median(s)
            hx_group = get_median_group(n, mid_points, j)
            Hx = (hx, hx_group)

            data[f'h{index}'] = Hx

        if moda:
            dx = get_moda(s, h)
            dx_group = get_moda_group(mid_points, j)
            Moda = (dx, dx_group)

            data[f'd{index}'] = Moda

    data, index = dict(), []

    if type(X[0]) is list:
        otr, j, h_interval = get_interval_stat(X[0], a=A[0], b=B[0], count=count[0])
        mid_points_x = get_mid_points(otr)

        otr, j, h_interval = get_interval_stat(X[1], a=A[1], b=B[1], count=count[1])
        mid_points_y = get_mid_points(otr)

        n = len(X)
        for i in range(n):
            get_params(X[i], count[i], data, A[i], B[i], index=i)

        if cov:
            Kxy = get_cov(*X)
            Kxy_group = get_cov_group(h, (X[0], X[1]), (mid_points_x, mid_points_y), (len(X[0]), len(X[1])), A, B)

            data['Kxy'] = (Kxy, Kxy_group)

        if p:
            S = 1

            pxy = get_pxy(X)
            pxy_group = get_pxy_gruop(X, count, h, A, B)

            data['pxy'] = (pxy, pxy_group)
    else:
        get_params(X, count, data, A, B)

    return list(data.values()), list(data.keys())


def task3(X, count, h=None, A=None, B=None, cov=True, p=True, mean=True, moda=True, variance=True, unbiased_variance=True, median=True):
    data, index = get_params(X, count, A, B, h, cov, p, mean, moda, variance, unbiased_variance, median)

    df = pd.DataFrame(data=data, index=index, columns=['Стат', 'Груп'])
    print_df(df)


def main():
    s = read_var('data.xlsx', 9)
    s.sort()
    count = 7

    h, h_otn = get_h(s)

    otr, j, h_interval = get_interval_stat(s, count)
    mid_points = get_mid_points(otr)
    interval_stat = (otr, j, h_interval)

    task1(s)
    task2(s, mid_points, interval_stat)
    task3(s, h, mid_points, j)


if __name__ == "__main__":
    main()
