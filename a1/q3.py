import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


BATCHES = 50
K = 500
np.random.seed(0)


class BatchSampler(object):
    """
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    """

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size
        self.indices = np.arange(self.num_points)
        self.X = data
        self.y = targets

    def random_batch_indices(self, m=None):
        """
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        """
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        """
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        """
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.X, indices, 0)
        y_batch = self.y[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    """
    Load the Boston houses dataset and randomly initialise linear regression weights.
    """
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    X = np.concatenate((np.ones((506, 1)), X), axis=1)
    features = X.shape[1]


    # Initialize w
    w = np.random.randn(features)
    print(w.shape)
    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity (cos theta) between two vectors.
    """
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


# implement linear regression gradient
def lin_reg_gradient(X, y, w):
    """
    Compute gradient of linear regression model parameterized by w
    """
    s = X.shape[0]
    gradient_w = - 2 * np.matmul(X.T, (y - np.matmul(X, w)))/s
    return gradient_w


# helper function for last question, calculate the variance matrix.
def calculate_gradient_var(batch_sampler, w, K):
    i = 0
    # init a 500 * 14 matrix
    gradient_matrix = np.zeros((K, batch_sampler.features), dtype = float)
    total_loss_gradient = 0
    while i < K:
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        gradient_matrix[i, :] = batch_grad
        total_loss_gradient += batch_grad
        i += 1
    gradient = np.divide(total_loss_gradient, K)
    # take variance for each column result is a (1,14) matrix
    vector_variance = np.log(cal_var(gradient_matrix))
    print("shape of vector variance is:")
    print(vector_variance.shape)
    return gradient, vector_variance


# calculate variance for a vector data.
def cal_var(g):
    mean = g.mean(0)
    var_vector = np.zeros((1, g.shape[1]), dtype=float)
    for i in range(g.shape[0]):
        var_vector += np.power(g[i, :] - mean, 2)
    result = np.divide(var_vector, (g.shape[0]))
    return result


# plot the graph.
def plot_delta_j_vs_m(X, y, w, m_start, m_end, K):
    m_range = m_end-m_start
    x_list = np.arange(1, m_range+1)
    x_list = np.log(x_list)
    x_list.tolist()
    variance_matrix = np.zeros((m_range, X.shape[1]), dtype=float)
    plt.figure(figsize=(20, 5))
    for i in range(1, m_range+1):
        batch_sampler = BatchSampler(X, y, i)
        gradient, vector_variance = calculate_gradient_var(batch_sampler, w, K)
        print("i is:")
        print(i)
        variance_matrix[i-1, :] = vector_variance

    for j in range(X.shape[1]):
        print("j is: ")
        print(j)
        # y to plot is jth column for the vector variance.
        print(variance_matrix.shape)
        y_list = variance_matrix[:, j]
        plt.subplot(5, 3, j + 1)
        plt.plot(x_list, y_list)
        plt.title("W_{}".format(j))
        plt.xlabel("Log M")
        plt.ylabel("Log Var")
    plt.tight_layout()
    plt.show()

    # plot only one graph.
    # y_list = variance_matrix[:, 4]
    # plt.plot(x_list, y_list)
    # plt.title("W_{}".format(4))
    # plt.xlabel("Log M")
    # plt.ylabel("Log Var")
    # plt.show()


def mse(predict_value, test_value):
    # print(predict_value, test_value)
    return np.mean(np.power(test_value-predict_value, 2))


def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    K = 500
    # Create a batch sampler to generate random batches from data
    print("shape y is:")
    print(y.shape)
    print("shape w is:")
    print(w.shape)
    print("shape x is:")
    print(X.shape)
    batch_sampler = BatchSampler(X, y, BATCHES)
    true_gradient = lin_reg_gradient(X, y, w)
    gradient_computed, vector = calculate_gradient_var(batch_sampler, w, K)
    print("true gradient is:")
    print(true_gradient)

    # Example usage
    MSE = mse(gradient_computed, true_gradient)
    cosine_similarity_loss = cosine_similarity(gradient_computed, true_gradient)

    print("MSE is:")
    print(MSE)
    print("Cosine similarity loss is:")
    print(cosine_similarity_loss)

    plot_delta_j_vs_m(X, y, w, 1, 401, K)


if __name__ == '__main__':
    main()

