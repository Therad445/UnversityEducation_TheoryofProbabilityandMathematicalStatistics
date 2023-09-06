import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as sp


def read_var(name, N):
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


def task1(s, count):
    items = list(set(s))
    items.sort()
    h, h_otn = get_h(s)

    stat = pd.DataFrame(data=[items, h, h_otn], index=['Значения', 'Частоты', 'Относительная частота'])
    var = pd.DataFrame(data=s, columns=['Значения']).T

    print('Вариационный ряд:')
    print_df(var)
    print('Статистический ряд:')
    print_df(stat)
    print(f'Размах выборки:\t{s[-1] - s[0]}')


def get_mid_points(s, count):
    a, b = s[0], s[-1]
    buf = np.linspace(a, b, count + 1)

    step = buf[1] - buf[0]
    return [a + step * i + step / 2 for i in range(0, count)]


def get_interval_stat(s, count):
    a, b = s[0], s[-1]
    n = len(s)

    # Разбиение выборки на отрезки
    buf = np.linspace(a, b, count + 1)
    otr = [(buf[i - 1], buf[i]) for i in range(1, len(buf))]

    # Нахождение частот
    k = 0
    j = [0 for i in range(count)]
    j[0] = 1
    for x in s:
        if otr[k][0] < x <= otr[k][1]:
            j[k] += 1
        elif x > otr[k][1]:
            k += 1
            j[k] += 1

    # Нахождение относительных частот
    h_interval = [c / n for c in j]
    return otr, j, h_interval


def plot_interval_stat(s, mid_points):
    count = len(mid_points)

    step = mid_points[1] - mid_points[0]
    points = [mid_points[i//2] + step/2 if i % 2 else mid_points[i//2] - step/2 for i in range(2*count)]

    h, h_otn = get_h(s)
    otr, j, h_interval = get_interval_stat(s, count)

    value = [sum(h_interval[0:(i//2 + 1)]) for i in range(2*count)]

    print(value)

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
    print(mid_points)
    print('Интервальный статистический ряд')
    print_df(interval_stat)

    # построение гистограммы частот
    plot_interval_stat(s, mid_points)


def task3(s, h, mid_points, j):
    count = len(mid_points)
    step = mid_points[1] - mid_points[0]
    n = len(s)

    mean = sum(s)/n
    mean_group = sum([mid_points[i]*j[i] for i in range(count)])/n

    D = sum([(s[i] - mean) ** 2 for i in range(n)])/n
    D_group = sum([(mid_points[i] - mean_group) ** 2 * j[i] for i in range(count)])/n

    S = n * D / (n - 1)
    S_group = n * D_group / (n - 1)

    i = n//2
    median = s[i + 1] if n % 2 else 1/2 * (s[i] + s[i + 1])
    i = count//2
    x_l = mid_points[i] if count % 2 else 1/2 * (mid_points[i + 1] + mid_points[i])

    print((x_l - step/2))
    print(sum([j[k] for k in range(i)]), n/2)
    print(step)
    median_group = (x_l - step/2) + ((n/2 - sum([j[k] for k in range(i)])) / j[i]) * step

    # moda
    moda = []
    # Стат
    for i in range(n):
        if h[i] == max(h):
            moda += [s[i]]

    # Груп
    max_indexes = []
    for i in range(count):
        if j[i] == max(j):
            max_indexes += [i]

    a_d = sum([mid_points[i] for i in max_indexes])/len(max_indexes) - step/2
    n_d = sum([j[i] for i in max_indexes])/len(max_indexes)
    n_d_1 = j[max_indexes[0] - 1]
    n_d_2 = j[max_indexes[-1] + 1]
    moda_group = a_d + ((n_d - n_d_1)/(2 * n_d - n_d_1 - n_d_2)) * step

    Mx = (mean, mean_group)
    Dx = (D, D_group)
    Sx = (S, S_group)
    hx = (median, median_group)
    Moda = (moda, moda_group)

    df = pd.DataFrame(data=[Mx, Dx, Sx, hx], index=['Mx', 'Dx', 'Sx', 'hx'], columns=['Стат', 'Груп'])
    print_df(df)
    print(f'Moda {Moda}')


def main():
    s = read_var('data.xlsx', 9)
    s.sort()
    count = 7

    h, h_otn = get_h(s)

    mid_points = get_mid_points(s, count)
    otr, j, h_interval = get_interval_stat(s, count)
    interval_stat = (otr, j, h_interval)

    task1(s, count)
    task2(s, mid_points, interval_stat)
    task3(s, h, mid_points, j)


main()
