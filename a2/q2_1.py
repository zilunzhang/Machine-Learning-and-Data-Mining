'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''
from sklearn.model_selection import KFold
import data
import numpy as np
import collections
from datetime import datetime
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        # print("shape of test point is: {}".format(test_point.shape))
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        l2_distance_list = self.l2_distance(test_point)
        # print("l2 norm list is: {}".format(l2_distance_list.shape))
        sort_indices = np.argsort(l2_distance_list)[:k]
        # changed
        sort_distance = np.take(l2_distance_list, sort_indices)[:k]
        # print("desired indices list is: {}".format(sort_indices.shape))
        desired_class_lables = np.take(self.train_labels, sort_indices)
        desired_class_lables = desired_class_lables.astype(int)
        # print("desired class label is: {}".format(desired_class_lables))


        list = desired_class_lables.tolist()
        # counts = np.bincount(desired_class_lables)
        # digit = np.argmax(counts)
        counter = collections.Counter(list)
        result = counter.most_common()
        max = result[0][1]
        index_list = []
        for tuple in result:
            if tuple[1] == max:
                index_list.append(tuple[0])
            else:
                break
        if k == 1 or len(index_list) == 1:
           return index_list[0]
        else:
            # print("recursion here")
            return self.query_knn(test_point, k-1)


#     def query_knn(self, test_point, k):
#         '''
#         Query a single test point using the k-NN algorithm
#
#         You should return the digit label provided by the algorithm
#         '''
#         l2_distance_list = self.l2_distance(test_point)
#         # print("l2 norm list is: {}".format(l2_distance_list.shape))
#         sort_indices = np.argsort(l2_distance_list)[:k]
#         # changed
#         sort_distance = np.take(l2_distance_list, sort_indices)[:k]
#         # print("desired indices list is: {}".format(sort_indices.shape))
#         desired_class_lables = np.take(self.train_labels, sort_indices)
#         desired_class_lables = desired_class_lables.astype(int)
#         # print("desired class label is: {}".format(desired_class_lables))
#
#
#         list = desired_class_lables.tolist()
#         # digit = get_max_distance(list, sort_distance)
#         digit = get_max_distance(list, sort_distance)
#         # counts = np.bincount(desired_class_lables)
#         # digit = np.argmax(counts)
#
#         return digit
#
#
# def get_max_distance(list, sort_distance):
#     # deal_with_tie basing on rank
#     counter = collections.Counter(list)
#     result = counter.most_common()
#     max = result[0][1]
#     index_list = []
#     for tuple in result:
#         if tuple[1] == max:
#             index_list.append(tuple[0])
#         else:
#             break
#
#     if len(index_list)==1:
#         return index_list[0]
#     else:
#         sum_list = []
#         for element in index_list:
#             element_index = list.index(element)
#             sum_list.append(np.sum(np.take(sort_distance, element_index)))
#         desired_index = np.argmin(sum_list)
#         desired_k = np.take(index_list, desired_index)
#         return desired_k


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    average_accuracy_cross_folds = []
    for k in k_range:
        print("k number is: {}".format(k))
        kf = KFold(n_splits=10, shuffle=False)
        print("split number is: {}".format(kf.get_n_splits(train_data)))
        test_accuracy = []
        fold_num = 0
        # Loop over folds
        for train_index, test_index in kf.split(train_data):
            fold_num += 1
            print("fold number is: {}".format(fold_num))
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            # Evaluate k-NN
            knn = KNearestNeighbor(X_train, y_train)
            accuracy = classification_accuracy(knn, k, X_test, y_test)
            test_accuracy.append(accuracy)
        average_accuracy_cross_folds.append(np.mean(test_accuracy))
    print("average accuracies are: {}".format(average_accuracy_cross_folds))
    opt_k = np.argmax(average_accuracy_cross_folds)
    return opt_k + 1, np.take(average_accuracy_cross_folds, opt_k)

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predict_label = []
    for test_data in eval_data:
        predict_label.append(knn.query_knn(test_data, k))

    num_total = len(eval_labels)
    temp_result = np.subtract(predict_label,  eval_labels)
    temp_result = temp_result.tolist()
    temp_count = temp_result.count(0)

    accuracy = temp_count/num_total

    return accuracy





def main():
    start = datetime.now()
    print("start time is: {}".format(start))
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    knn = KNearestNeighbor(train_data, train_labels)

    one_accuracy_train = classification_accuracy(knn, 1, train_data, train_labels)
    fifteen_accuracy_train = classification_accuracy(knn, 15, train_data, train_labels)

    one_accuracy_test = classification_accuracy(knn, 1, test_data, test_labels)
    fifteen_accuracy_test = classification_accuracy(knn, 15, test_data, test_labels)

    print("train accuracy when k = 1 is: {}".format(one_accuracy_train))
    print("train accuracy when k = 15 is: {}".format(fifteen_accuracy_train))
    print("test accuracy when k = 1 is: {}".format(one_accuracy_test))
    print("test accuracy when k = 15 is: {}".format(fifteen_accuracy_test))

    optimal_k, opt_ave_acc = cross_validation(train_data, train_labels, k_range=np.arange(1,16))

    print("optimal k is: {}".format(optimal_k))

    print("average accuracy for k is: {}".format(opt_ave_acc))

    test_acc = classification_accuracy(knn, optimal_k, test_data, test_labels)

    print("test accuracy for optimal k is: {}".format(test_acc))

    stop = datetime.now()

    print("run time is: {}".format(stop-start))

if __name__ == '__main__':
    main()