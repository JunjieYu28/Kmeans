import os
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load(os.path.join('data', 'data_example.npy'))
label = np.load(os.path.join('data', 'label_example.npy'))
print(data.shape, label.shape)

plt.scatter(data[:, 0], data[:, 1], c=label)
plt.axis('equal')
plt.savefig('data_example_.jpg')

print([(data[i], label[i]) for i in range(10)])
