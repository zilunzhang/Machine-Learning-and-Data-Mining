import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0, velocity = 0.0):
        self.lr = lr
        self.beta = beta
        self.vel = velocity

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        delta = -grad * self.lr + self.beta * self.vel
        params += delta
        self.vel = delta
        return  params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        # self.w = np.zeros((feature_count, 1)).ravel()


    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss

        hinge_loss =  1 - y * np.matmul(X, self.w)
        hinge_loss[hinge_loss < 0] = 0

        return hinge_loss


    # def objective_loss (self, X,y):
    #
    #     regularizer_w = self.w[1:]
    #     regularizer = 1/2 * np.dot(regularizer_w, (regularizer_w).T)
    #     hinge_loss = self.hinge_loss(X, y)
    #     objective_loss = regularizer + self.c * np.mean(hinge_loss)
    #
    #     return objective_loss




    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective

        # fake_hinge_loss = 1 - y * np.matmul(X, self.w)
        # desired_indices = np.argwhere(fake_hinge_loss <= 0).ravel()
        # y[desired_indices] = 0
        # # X[desired_indices, :] = 0
        # reg_w = self.w
        # reg_w[0] = 0
        # temp_gra = np.matmul(X.T, - y)
        # grad = reg_w + self.c/np.count_nonzero(y) * temp_gra
        # return grad

        regularizer = self.w
        regularizer[0] = 0

        grad = 0
        for (x_i, y_i) in zip(X, y):
            fake_hinge_loss = 1 - y_i * np.matmul(self.w, x_i)
            if fake_hinge_loss > 0:
                grad += - y_i * x_i
            else:
                grad += 0

        return regularizer + (self.c / X.shape[0]) * grad


    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.sign(np.matmul(X, self.w))


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    print("Train data shape: {}".format(train_data.shape))
    print("Test data shape: {}".format(test_data.shape))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
    x = np.arange(201, step=1)
    draw_gd(x, w_history)
    print("Final prediction for {} steps is: {}".format(steps, w_history[-1]))
    return w_history


def draw_gd(x, y):
    plt.plot(x, y)
    plt.title("Steps V.S. Function Value")
    plt.xlabel("Steps")
    plt.ylabel("Function Value")


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'

    update to svm weight
    '''
    svm = SVM(c=penalty, feature_count=train_data.shape[1])
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)

    w_init = svm.w
    w_history = [w_init]
    hinge_loss_matrix = np.zeros((iters, batchsize))
    for i in range(iters):
        print("iteration number is: {}".format(i))
        X_batch, y_batch = batch_sampler.get_batch()
        loss = svm.hinge_loss(X_batch, y_batch)
        hinge_loss_matrix[i, :] = loss
        # Optimize and update the history
        svm.w = optimizer.update_params(svm.w, svm.grad(X_batch, y_batch))
        w_history.append(svm.w)
    ave_hinge_loss = np.mean(hinge_loss_matrix[-1, :])
    return w_history, ave_hinge_loss, svm


def plot_w(w):
    image = np.reshape(w, (28, 28))
    plt.imshow(image, cmap='gray')
    plt.title("w")
    plt.show()


def plot_loss(loss):
    x = np.arange(len(loss))
    plt.plot(x, loss)
    plt.title("Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':


    # GDO_1 = GDOptimizer(lr = 1.0, beta=0.0, velocity = 0.0)
    # w_history_1 = optimize_test_function(GDO_1)
    # GDO_2 = GDOptimizer(lr = 1.0, beta=0.9, velocity = 0.0)
    # w_history_2 = optimize_test_function(GDO_2)
    # plt.legend(["with beta = 0.0", "with beta = 0.9"])
    # plt.show()


    m = 100
    T = 500
    alpha = 0.05
    c = 1.0
    beta_1 = 0.0
    beta_2 = 0.1
    train_data, train_targets, test_data, test_targets = load_data()

    train_data = np.concatenate((np.ones((train_data.shape[0], 1)), train_data), axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0], 1)), test_data), axis=1)

    feature_count = train_data.shape[1]

    GDO_SVM_1 = GDOptimizer(lr=alpha, beta=beta_1, velocity=0.0)
    GDO_SVM_2 = GDOptimizer(lr=alpha, beta=beta_2, velocity=0.0)


    w_history_1, train_loss_1_no, svm_1 = optimize_svm(train_data, train_targets, c, GDO_SVM_1, 100, 500)

    train_predict = svm_1.classify(train_data)

    train_acc = (train_predict == train_targets).mean()

    test_predict = svm_1.classify(test_data)

    test_acc = (test_predict == test_targets).mean()

    train_loss_1 = svm_1.hinge_loss(train_data, train_targets)

    test_loss_1 = svm_1.hinge_loss(test_data, test_targets)

    print("train accuracy is: {}, train loss is: {}".format(train_acc, np.mean(train_loss_1)))

    print("test accuracy is: {}, test loss is: {}".format(test_acc, np.mean(test_loss_1)))

    # plot_w(w_history_1[-1][1:])




    # w_history_2, train_loss_no, svm_2 = optimize_svm(train_data, train_targets, c, GDO_SVM_2, 100, 500)
    #
    # train_predict = svm_2.classify(train_data)
    #
    # train_acc = (train_predict == train_targets).mean()
    #
    # test_predict = svm_2.classify(test_data)
    #
    # test_acc = (test_predict == test_targets).mean()
    #
    #
    # train_loss_2 = svm_2.hinge_loss(train_data, train_targets)
    #
    # test_loss_2 = svm_2.hinge_loss(test_data, test_targets)
    #
    # print("train accuracy is: {}, train loss is: {}".format(train_acc, np.mean(train_loss_2)))
    #
    # print("test accuracy is: {}, test loss is: {}".format(test_acc, np.mean(test_loss_2)))
    #
    # plot_w(w_history_2[-1][1:])