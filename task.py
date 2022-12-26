# import numpy
import numpy as np 

# import SVM model
from sklearn import svm

# import syn-data generator
from sklearn.datasets import make_moons, make_blobs

# import visulization tool
from matplotlib import pyplot as plt
plt.style.use('ggplot')



# task configuration
n_samples = 800
abnormal_rate = 0.25
n_abnormal = int(n_samples * abnormal_rate)
n_normal = n_samples - n_abnormal 
random_seed = 42

model = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
name = "One-Class SVM"

# generate synthetic dataset (normal)
blobs_params = dict(random_state=0, n_samples=n_normal, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0]], cluster_std=[1],
               **blobs_params)[0],
    make_blobs(centers=[[-2, 0], [2, 0]], cluster_std=[1, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.3, 0.6],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25]))
    ]

# grid points for drawing contours
xx, yy = np.meshgrid(np.linspace(-7, 7, 300),
                     np.linspace(-7, 7, 300))

# initialize figure
plt.figure(figsize=(10, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)


plot_num = 1

for i_dataset, X in enumerate(datasets):
    
    # training model without abnormal data
    model.fit(X)

    plt.subplot(len(datasets), 2, plot_num)
    if i_dataset == 0:
        plt.title(name, size=18)

    # generate abnormal data
    abnormal = np.random.RandomState(random_seed).uniform(low=-6, high=6,
                       size=(n_abnormal, 2))
    # predict data with abnormal data added
    X_concat = np.concatenate([X, abnormal], axis=0)
    y_pred = model.predict(X_concat)


    # plot the levels lines
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=3, colors='black')

    # plot the normal and abnormal points
    colors = np.array(['#377eb8', '#ff7f00'])
    plt.scatter(X_concat[:, 0], X_concat[:, 1], s=10, color=colors[(y_pred + 1) // 2])

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)


    plot_num += 1
    
    # plot the grund truth points
    plt.subplot(len(datasets), 2, plot_num)
    if i_dataset == 0:
        plt.title("Original Data", size=18)

    
    plt.scatter(X[:, 0], X[:, 1], s=10, color='#ff7f00', label="Normal")
    plt.scatter(abnormal[:, 0], abnormal[:, 1], s=10, color='#377eb8', label="Abnormal")
    

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.legend()

    plot_num += 1

#plt.show()
plt.savefig("output_figure.png")