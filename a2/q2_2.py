'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from datetime import datetime


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        mean_digit = np.mean(i_digits, 0)
        means[i, :] = mean_digit
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    means = compute_mean_mles(train_data, train_labels)
    covariances = np.zeros((10, 64, 64))
    result = []
    # Compute covariances
    for q in range(0, 10):
        print("covariance's iteration is :{}".format(q + 1))
        i_digits = data.get_digits_by_label(train_data, train_labels, q)
        # Compute mean of class i
        shift = (i_digits - means[q])
        covariance = np.divide(np.matmul(shift.T, shift), i_digits.shape[0]-1)
        covariance += 0.01 * np.identity(covariance.shape[0])
        covariances[q, :, :] = covariance
        result.append(covariances[q])
    all_concat = np.concatenate(result, 0)
    np.savetxt("q2 cov.csv", all_concat, delimiter=",")
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    sigmas = []
    for i in range(10):
        # len: 64
        cov_diag = np.diag(covariances[i])
        log_diag = np.log(cov_diag)
        # shape 8*8
        log_diag = np.reshape(log_diag, (8,8))
        sigmas.append(log_diag)
    # shape 8*8*10
    all_concat = np.concatenate(sigmas, 1)
    np.savetxt("q2 covariances.csv", all_concat, delimiter=",")
    print("concat good!")
    plt.imshow(all_concat, cmap='gray')
    print("imshow good!")
    plt.show()
    print("all good!")


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = digits.shape[0]
    d = digits.shape[1]
    return_matrix = np.zeros((n, 10))
    log_two_pi_d = -0.5 * d * np.log(2 * np.pi)
    # for each point x
    for i in range(0, 10):
        print("generative likelihood iteration is: {}".format(i + 1))
        det = np.linalg.det(covariances[i])
        inverse = np.linalg.inv(covariances[i])
        det_stuff = -0.5 * np.log(det)
        former = log_two_pi_d + det_stuff
        for j in range(n):
            shift = digits[j]-means[i]
            content = np.dot(np.dot(shift.T, inverse), shift)
            latter = -0.5 * content
            p = former + latter
            return_matrix[j,i] = p
    np.savetxt("q2 generative likelihood.csv", np.exp(return_matrix), delimiter=",")
    np.savetxt("q2 generative log likelihood.csv", return_matrix, delimiter=",")
    return return_matrix


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma) = log p(x|y,mu,Sigma) + log p(y|mu, sigma) - log p(x|mu, sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of data points and 10 corresponds to each digit class
    '''
    log_g_l = generative_likelihood(digits, means, covariances)

    log_p_y = np.log(1/10)

    log_prior_likelihood = log_g_l + log_p_y

    p_x_vector = np.sum(np.exp(log_prior_likelihood), axis=1)

    p_x = np.tile(p_x_vector.T, (log_g_l.shape[1], 1)).T

    log_p_x = np.log(p_x)

    c_l = log_prior_likelihood - log_p_x

    np.savetxt("q2 conditional log likelihood.csv", c_l, delimiter=",")

    return c_l


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    correct_conditional_likelihood = []
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for i in range(digits.shape[0]):
        correct_conditional_likelihood.append(cond_likelihood[i][int(labels[i])])
    ave_l = np.mean(correct_conditional_likelihood)
    return ave_l


def classify_data(digits, means, covariances, labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    most_p_digits = np.argmax(cond_likelihood, axis=1)
    num_total = len(labels)
    temp_result = np.subtract(most_p_digits,  labels)
    temp_result = temp_result.tolist()
    temp_count = temp_result.count(0)
    accuracy = temp_count/num_total
    return accuracy


def main():

    start = datetime.now()
    print("start time is: {}".format(start))

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    print("train data's shape is: {}".format(train_data.shape))

    print("train labels' shape is: {}".format(train_labels.shape))

    print("test data's shape is: {}".format(test_data.shape))

    print("test labels' shape is: {}".format(test_labels.shape))

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    np.savetxt("q2 means.csv", means, delimiter=",")
    print("means' shape is: {}".format(means.shape))

    covariances = compute_sigma_mles(train_data, train_labels)
    print("sigma's shape is: {}".format(covariances.shape))
    plot_cov_diagonal(covariances)

    # g_l, log_g_l = generative_likelihood(train_data, means, covariances)
    # print("generative likelihood's shape is: {}".format(log_g_l))

    # c_l = conditional_likelihood(train_data, means, covariances)
    # print("conditional likelihood's shape is: {}".format(c_l.shape))


    train_ave_l = avg_conditional_likelihood(train_data, train_labels, means, covariances)


    test_ave_l = avg_conditional_likelihood(test_data, test_labels, means, covariances)


    # Evaluation

    train_classify_data = classify_data(train_data, means, covariances, train_labels)

    test_classify_data = classify_data(test_data, means, covariances, test_labels)

    print("train average log likelihood is: {}".format(train_ave_l))

    print("train average probability is: {}".format(np.exp(train_ave_l)))

    print("test average log likelihood is: {}".format(test_ave_l))

    print("test average probability is: {}".format(np.exp(test_ave_l)))

    print("train accuracy is: {}".format(train_classify_data))

    print("test accuracy is: {}".format(test_classify_data))

    stop = datetime.now()

    print("run time is: {}".format(stop-start))

if __name__ == '__main__':
    main()
