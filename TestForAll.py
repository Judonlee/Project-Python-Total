import numpy


def catch(a, b):
    x = (a, b)
    return a, x


if __name__ == '__main__':
    x, y = catch(1, 2)
    # print(x, y)
    print(y[1])
