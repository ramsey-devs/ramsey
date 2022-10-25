from jax import grad


def g(x):
    return x * 2


def h(x):
    return x * x


def f(x):
    if x > 2.0:
        return 2.0 * x * g(x)
    return x * h(x)


f_prime = grad(f)
s = f_prime(2.4)
print(s)

s = f_prime(2.0)
print(s)
