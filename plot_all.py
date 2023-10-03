import numpy as np
import matplotlib.pyplot as plt
import glob

res = glob.glob('data/2*.npy')
for i in res:
    henk = np.load(i)[::-1]
    plt.imshow(henk, cmap='gray')
    plt.title(i)
    plt.show()

    plt.hist(henk.flatten())
    plt.show()