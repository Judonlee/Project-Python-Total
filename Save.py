import random

if __name__ == '__main__':
    x, y = 4.619, 5.955
    print('%.2f\t%.2f' % (x - random.random() + 0.6, y + random.random() - 0.6))
