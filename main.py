import numpy as np
import matplotlib.pyplot as plt

L = 3.5
N = [3, 10, 30]


def frequency(t, n):
    return t * np.pi * n / 7


def fourier_sum(x, N, a0, a_func=None, b_func=None, w_factor=2):
    s = np.full_like(x, a0 / 2, dtype=float)

    for n in range(1, N + 1):
        c = frequency(w_factor, n)

        if a_func is not None:
            s += a_func(n) * np.cos(c * x)

        if b_func is not None:
            s += b_func(n) * np.sin(c * x)

    return s


def plot_series(x, partial_func, ref_func, title):
    plt.figure(figsize=(10, 5))

    for n in N:
        plt.plot(x, partial_func(x, n), label=f"N={n}")

    plt.plot(x, ref_func(x), linestyle="--", linewidth=2, label="сумма ряда")

    plt.xlabel("x")
    plt.ylabel("S_N(x)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.show()


def a_common(n):
    c = frequency(4, n)

    return -(np.sin(c) + np.sin(2 * c) + np.sin(3 * c)) / (np.pi * n)


def b_common(n):
    c = frequency(4, n)

    return (np.cos(c) + np.cos(2 * c) + np.cos(3 * c) - 3) / (np.pi * n)


def a_even(n):
    c = frequency(2, n)

    return -(2 / (np.pi * n)) * (np.sin(c) + np.sin(2 * c) + np.sin(3 * c))


def b_odd(n):
    c = frequency(2, n)

    return (2 / (np.pi * n)) * (
        np.cos(c) + np.cos(2 * c) + np.cos(3 * c) - 3 * ((-1) ** n)
    )


def S_common(x, N):
    return fourier_sum(
        x=x, N=N, a0=18 / 7, a_func=a_common, b_func=b_common, w_factor=4
    )


def S_even(x, N):
    return fourier_sum(x=x, N=N, a0=18 / 7, a_func=a_even, b_func=None, w_factor=2)


def S_odd(x, N):
    return fourier_sum(x=x, N=N, a0=0, a_func=None, b_func=b_odd, w_factor=2)


def sum_common_ref(x):
    return np.floor(np.mod(x, L))


def sum_even_ref(x):
    y = np.mod(x + L, 2 * L) - L
    return np.floor(np.abs(y))


def sum_odd_ref(x):
    y = np.mod(x + L, 2 * L) - L
    return np.sign(y) * np.floor(np.abs(y))


x_common = np.linspace(-3.5, 7, 3000)
x_ext = np.linspace(-7, 7, 4000)

plot_series(x_common, S_common, sum_common_ref, "Общий тригонометрический ряд")

plot_series(x_ext, S_even, sum_even_ref, "Косинусный ряд (четное продолжение)")

plot_series(x_ext, S_odd, sum_odd_ref, "Синусный ряд (нечётное продолжение)")
