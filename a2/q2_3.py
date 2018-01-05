'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from datetime import datetime


def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5

    return the data which > 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))

    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        i_digits = np.concatenate((np.ones((1, i_digits.shape[1])), i_digits), axis=0)
        i_digits = np.concatenate((np.zeros((1, i_digits.shape[1])), i_digits), axis=0)
        # print(i_digits.shape)
        binary_digits = binarize_data(i_digits)
        probability_digits = np.mean(binary_digits,axis= 0)
        eta[i, :] = probability_digits
    return eta


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    container = []
    for i in range(10):
        img_i = class_images[i]
        temp = np.reshape(img_i, (8, 8))
        container.append(temp)
    all_concat = np.concatenate(container, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(0, 10):
        uniform_rv = np.random.uniform(0.0, 1.0, (1, 64))
        generated_data [i, :]= np.where(uniform_rv > eta[i], 1.0, 0.0)
    plot_images(generated_data)
    np.savetxt("q3 generated.csv", generated_data, delimiter=",")
    return generated_data


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''

    n = bin_digits.shape[0]
    # d = bin_digits.shape[1]
    return_matrix = np.zeros((n, 10))

    for j in range(0,10):
        print("generative likelihood iteration is: {}".format(j + 1))
        for k in range(n):
            p = np.power(eta[j], bin_digits[k]) * np.power(1-eta[j], 1-bin_digits[k])
            p = np.prod(p)
            return_matrix [k, j] = p

    return np.log(return_matrix)


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta) = log p(x|y,eta) + log(1/10) - log p(x|eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of data points and 10 corresponds to each digit class
    '''

    log_g_l = generative_likelihood(bin_digits, eta)

    # np.savetxt("q3 generative likelihood.csv", g_l, delimiter=",")

    np.savetxt("q3 generative log likelihood.csv", log_g_l, delimiter=",")

    g_l = np.exp(log_g_l)

    g_l = g_l * 0.1

    p_x = np.sum(g_l, axis=1)

    p_x_matrix = np.tile(p_x.T, (10, 1)).T

    log_p_x = np.log(p_x_matrix)

    c_l = log_g_l + np.log(1/10) - log_p_x

    np.savetxt("q3 conditional log likelihood.csv", c_l, delimiter=",")

    return c_l

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    correct_conditional_likelihood = []
    for i in range(bin_digits.shape[0]):
        correct_conditional_likelihood.append(cond_likelihood[i][int(labels[i])])
    ave_l = np.mean(correct_conditional_likelihood)
    return ave_l

def classify_data(bin_digits, eta, labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)


    np.savetxt("eta matrix.csv", eta, delimiter=",")

    # Evaluation
    plot_images(eta)

    generated_data = generate_new_data(eta)

    np.savetxt("generated data.csv", generated_data, delimiter=",")

    print("generated data's shape is: {}".format(generated_data.shape))

    train_ave_l = avg_conditional_likelihood(train_data, train_labels, eta)

    test_ave_l = avg_conditional_likelihood(test_data, test_labels, eta)

    # Evaluation

    train_classify_data = classify_data(train_data, eta, train_labels)

    test_classify_data = classify_data(test_data, eta, test_labels)

    print("train average log likelihood is: {}".format(train_ave_l))

    print("train average probability is: {}".format(np.exp(train_ave_l)))

    print("test average log likelihood is: {}".format(test_ave_l))

    print("test average probability is: {}".format(np.exp(test_ave_l)))

    print("train accuracy is: {}".format(train_classify_data))

    print("test accuracy is: {}".format(test_classify_data))

    stop = datetime.now()

    print("run time is: {}".format(stop - start))

    print("Summary:" + "\n" + "KNN: best when k = 4, average accuracy is 0.9656, test accuracy is 0.9728." + "\n"
          + "Gaussian Naive Bayes: train accuracy is 0.9814, test accuracy is 0.9728." + "\n" +
          "Bernoulli Naive Bayes: train accuracy is 0.7741, test accuracy is 0.7643." + "\n" +
          "Best is Gaussian Naive Bayes, and worst is Bernoulli Naive Bayes." + "\n" +
          "That matches my expectation "
          "because the Handwritten Digit Classification should follow a gaussian distribution, instead of a bernoulli distribution. " + "\n" +
          "So that Gaussian is the best and the Bernoulli is the worst, since Gaussian models the Handwritten Digit Classification best.  " + "\n" +
          "KNN is discriminat method instead of generative method, "
          "it should worse than best generative model but better than worst generative mode, which is in between.")


if __name__ == '__main__':
    main()
