import numpy as np
import pathlib
import matplotlib.pyplot as plt


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    for i in range(9):  
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    
    plt.show()
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]

    return images, labels
