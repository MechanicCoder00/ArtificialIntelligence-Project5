# from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# files.upload()  # For Google Colab


def convolution2D(image2D, kernel3x3):
    print(image2D.shape)
    x, y = image2D.shape

    convolved2D = np.zeros((x, y))
    x = x - 2
    y = y - 2
    for i in range(x):
        for j in range(y):
            convolved2D[i][j] = np.sum(image2D[i:i + 3, j:j + 3] * kernel3x3)
    return convolved2D


image2D = np.loadtxt('one-channel.csv', delimiter=',')
sns.heatmap(image2D, cmap='Greens')
plt.show()

edge_detect_filter_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Convolve once
convolved_image = convolution2D(image2D, edge_detect_filter_3x3)

sns.heatmap(convolved_image, cmap='Greens')
plt.show()

# Convolve again
convolved_image = convolution2D(convolved_image, edge_detect_filter_3x3)

sns.heatmap(convolved_image, cmap='Greens')
plt.show()
