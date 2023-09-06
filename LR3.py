import LR1
import numpy as np
import scipy.stats as sp


def get_stat(s: list, count: int):
    print('----------------------------------------------------------------------')
    print(LR1.get_stat(s))

    otr, j, h_interval = LR1.get_interval_stat(s, count)

    a, b = otr[0][0], otr[-1][1]
    step = otr[0][1] - otr[0][0]
    mid_points = [a + i * step - step / 2 for i in range(1, count + 1)]

    m = LR1.get_mean(s)
    S = LR1.get_S_squared(len(s), LR1.get_D(s, m))

    print(f'm = {m}')
    print(f'S = {S ** (1 / 2)}')

    LR1.plot_interval_stat(s, mid_points)
    print('----------------------------------------------------------------------')


def main():
    def print_bounds(s):
        print('Границы негруппированной выборки:')

        mean = LR1.get_mean(s)
        var = LR1.get_D(s, mean)
        S = LR1.get_S_squared(50, var)

        i1 = sp.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=sp.sem(s))
        print(f'mx {i1}')

        i1 = ((50 - 1)*S/sp.chi2(50 - 1).ppf(0.975), (50 - 1)*S/sp.chi2(50 - 1).ppf(0.025))
        print(f'sig {i1[0], i1[1]}')

        print('Границы группированной выборки:')

        otr, j, h_interval = LR1.get_interval_stat(s, 7)
        mid = LR1.get_mid_points(otr)

        mean = LR1.get_group_mean(mid, j, 50)
        var = LR1.get_D_gruop(mid, j, 50, mean)
        S = LR1.get_S_squared(50, var)**(1/2)

        i2 = sp.t.interval(confidence=0.95, df=len(mid) - 1, loc=mean, scale=S)
        print(f'm {i2}')

        i2 = ((50 - 1) * S ** 2 / sp.chi2(50 - 1).ppf(0.975), (50 - 1) * S **2 / sp.chi2(50 - 1).ppf(0.025))
        print(f'sig {i2[0], i2[1]}')

    def task():
        s = LR1.read_var('data.xlsx', 9, 'Y', LR=2)
        s = sorted(s[1:len(s)])
        print(LR1.get_stat(s))
        LR1.task3(s, count, A=s[0], B=s[-1], cov=False, p=False, moda=False, median=False)

        print_bounds(s)

        n = sorted(np.random.normal(loc=N, size=size))
        r = sorted(np.random.uniform(low=a, high=b, size=size))
        e = sorted(np.random.exponential(scale=l, size=size))

        get_stat(n, count)
        get_stat(r, count)
        get_stat(e, count)

    N = 9
    size = 200
    count = 7

    a, b = N, 2*N
    l = N

    task()


main()


