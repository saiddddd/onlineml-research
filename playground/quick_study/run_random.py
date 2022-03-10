import numpy as np
import pandas as pd

import random

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
y = []
for i in range(10000):

    x1.append(random.gauss(100, 50))
    x2.append(random.gauss(100, 50))
    x3.append(random.gauss(100, 50))
    x4.append(random.gauss(100, 50))
    x5.append(random.gauss(100, 50))
    x6.append(random.gauss(100, 50))

    def get_y(cut_point_mean=100., criteria=0):

        cut_point = random.gauss(cut_point_mean, 3)

        if criteria == 0:
            if (x1[-1] > cut_point and x2[-1] < cut_point) or (
                    x1[-1] < cut_point and x3[-1] > cut_point):
                return 1
            else:
                return 0

        if criteria == 1:
            if (x1[-1] > cut_point and x2[-1] < cut_point) or (
                    x1[-1] < cut_point and x6[-1] > cut_point):
                return 1
            else:
                return 0

        if criteria == 2:
            if (x4[-1] > cut_point and x5[-1] < cut_point) or (
                    x4[-1] < cut_point and x6[-1] > cut_point):
                return 1
            else:
                return 0


    # target y1 generation heuristic
    # x1 > 40 and x2 < 30 or x1 < 20 and x2 > 60

    if i < 5000:
        y.append(get_y(criteria=0))
    else:
        y.append(get_y(cut_point_mean=50+0.01*i, criteria=0))

    #target y2 generation heuristic
    # x5 > 80 and x6 < 100 or x5 < 70 and x6 > 110

data = pd.DataFrame({
    "X1": x1,
    "X2": x2,
    "X3": x3,
    "X4": x4,
    "X5": x5,
    "X6": x6,
    "Y": y
    
})

#import pandas as pd
#pd.DataFrame()

data.to_csv("dummy_toy/dummy_data.csv", index=False)


import matplotlib.pyplot as plt

plt.hist(x1, bins=50, density=True, alpha=0.75)
plt.title('Random Variables distribution')
plt.xlabel('random variable')
plt.ylabel('A.U.')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.grid(True)
# plt.show()
plt.savefig('./random_variable_dist.pdf')
