import numpy as np
x = np.array([1, 2, 4, 8, 12, 16, 20, 24])
y = np.array([1, 2, 4, 6.8, 6.2, 9.8, 11.8, 14.5])
import matplotlib.pyplot as plt

if __name__ == '__main__':
    z = np.polyfit(x, y, 1)
    print(z)
    p = np.poly1d(z)
    xp = np.linspace(0, 26, 100)
    _ = plt.plot(x, y, '.', xp, p(xp), '-')
    plt.xlabel("threads") 
    plt.ylabel("speedup") 
    plt.title("y = 0.5448167x + 1.08761835")
    plt.ylim(0,16)
    plt.show()
