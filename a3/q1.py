'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test


def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    shape_test = bow_test.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('{} test data points.'.format(shape_test[0]))
    print('{} feature dimension.'.format(shape_test[1]))
    # print('{} label shape.'.format(len(feature_names)))

    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)

    shape = tf_idf_train.shape
    shape_test = tf_idf_test.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('{} test data points.'.format(shape_test[0]))
    print('{} feature dimension.'.format(shape_test[1]))

    print('Most common word in training set is "{}"'.format(feature_names[tf_idf_train.sum(axis=0).argmax()]))

    return tf_idf_train, tf_idf_test, feature_names


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    bnb = BernoulliNB()
    bnb.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_predict = bnb.predict(binary_train)
    train_acc =  (train_predict == train_labels).mean()
    print('BernoulliNB baseline train accuracy = {}'.format(train_acc))
    test_predict = bnb.predict(binary_test)
    test_acc = (test_predict == test_labels).mean()
    print('BernoulliNB baseline test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("bnb_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix


def gnb(train_data, train_labels, test_data, test_labels, hyper_parameter_info = None):

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)

    train_predict = gnb.predict(train_data)
    train_acc = (train_predict == train_labels).mean()
    print('GaussianNB train accuracy = {}'.format(train_acc))
    test_predict = gnb.predict(test_data)
    test_acc = (test_predict == test_labels).mean()
    print('GaussianNB test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("gnb_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix


def knn(train_data, train_labels, test_data, test_labels, k):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)

    train_predict = knn.predict(train_data)
    train_acc = (train_predict == train_labels).mean()
    print('KNN train accuracy = {}'.format(train_acc))
    test_predict = knn.predict(test_data)
    test_acc = (test_predict == test_labels).mean()
    print('KNN test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("knn_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix


def svm(train_data, train_labels, test_data, test_labels, hyperparameter_index):
    # kernel_list = ["linear", "poly", "rbf", "sigmoid"]
    # svm = SVC(class_weight="balanced", C =  penalize_coeff_list[hyperparameter_index], kernel= "linear")
    svm = LinearSVC(random_state=0, class_weight= "balanced", C=hyperparameter_index)
    svm.fit(train_data, train_labels)

    train_predict = svm.predict(train_data)
    train_acc = (train_predict == train_labels).mean()
    print('SVM train accuracy = {}'.format(train_acc))
    test_predict = svm.predict(test_data)
    test_acc = (test_predict == test_labels).mean()
    print('SVM test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("svm_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix


def random_forest(train_data, train_labels, test_data, test_labels, hyper):
    rf = RandomForestClassifier(random_state=1, n_jobs= -1, n_estimators= hyper)
    rf.fit(train_data, train_labels)
    train_predict = rf.predict(train_data)
    train_acc = (train_predict == train_labels).mean()
    print('Random Forest train accuracy = {}'.format(train_acc))
    test_predict = rf.predict(test_data)
    test_acc = (test_predict == test_labels).mean()
    print('Random Forest test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("random_forest_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix


def mnb(train_data, train_labels, test_data, test_labels, hyper_parameter_info):

    mnb = MultinomialNB(alpha=hyper_parameter_info, fit_prior=True)
    mnb.fit(train_data, train_labels)

    train_predict = mnb.predict(train_data)
    train_acc = (train_predict == train_labels).mean()
    print('MultinomialNB train accuracy = {}'.format(train_acc))
    test_predict = mnb.predict(test_data)
    test_acc = (test_predict == test_labels).mean()
    print('MultinomialNB test accuracy = {}'.format(test_acc))
    conf_matrix = confusion_matrix(test_predict, test_labels)
    np.savetxt("mnb_confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    return train_acc, test_acc, conf_matrix



def confusion_matrix(test_label, test_predict):
    # print("calculating confusion matrix......")
    confusion_matrix = np.zeros((20, 20))
    for i  in range(len(test_label)):
        predict_class = test_predict[i]
        # print(type(predict_class))
        label_class = test_label[i]
        # print(type(label_class))
        confusion_matrix[predict_class, label_class] += 1
    # print("done!")
    return confusion_matrix


def do_k_fold(train_data, train_labels, test_data, test_labels, model_name, hyper_parameter_info_list):

    if model_name == "knn" or "gnb" or "svm":
        pass
    else:
        raise NameError(" Not a valid name for model")

    # train_data = train_data.todense()
    # test_data = test_data.todense()

    average_accuracy_cross_folds = []
    for i in range(len(hyper_parameter_info_list)):
        print("k-fold loop index is: {}".format(i))
        kf = KFold(n_splits=10, shuffle=False)
        print("split number is: {}".format(kf.get_n_splits(train_data)))
        train_acc_list = []
        test_acc_list = []
        fold_num = 0
        # Loop over folds
        for train_index, test_index in kf.split(train_data):
            fold_num += 1
            print("fold number is: {}".format(fold_num))
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            # Evaluate model
            if model_name == "knn":
                train_acc, test_acc, conf_matrix = knn(X_train, y_train, X_test, y_test, hyper_parameter_info_list[i])
            elif model_name == "gnb":
                train_acc, test_acc, conf_matrix = gnb(X_train, y_train, X_test, y_test)
            elif model_name == "svm":
                train_acc, test_acc, conf_matrix = svm(X_train, y_train, X_test, y_test, hyper_parameter_info_list[i])
            elif model_name == "rf":
                train_acc, test_acc, conf_matrix = random_forest(X_train, y_train, X_test, y_test, hyper_parameter_info_list[i])
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        average_accuracy_cross_folds.append(np.mean(test_acc_list))
    print("average accuracies are: {}".format(average_accuracy_cross_folds))
    opt_index = np.argmax(average_accuracy_cross_folds)
    return hyper_parameter_info_list[opt_index], np.take(average_accuracy_cross_folds, opt_index)




def main():
    start = datetime.now()
    print("start time is: {}".format(start))

    train_data, test_data = load_data()
    # train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_bow, test_bow, feature_names = tf_idf_features(train_data, test_data)

    # best_k_index, best_test_accuracy = do_k_fold(train_bow, train_data.target, test_bow, test_data.target, "rf", np.arange(100, 201, 10))
    # # print("KNN's best k is: {}, with test acc: {}".format(best_k_index, best_test_accuracy))
    # print("SVM's best C is: {}, with test acc: {}".format(best_k_index, best_test_accuracy))
    # print("Random Forest's best k is: {}, with test acc: {}".format(best_k_index, best_test_accuracy))

    # result from k fold, k = 10:

    # KNN:
    # best KNN's k is 1.

    # SVM:
    # best Kernel is "linear"
    # best C (penelize term) is 1

    # Random Forest:
    # best number of estimator is 500

    # Multinomial Naive Bayes:
    # best alpha is 0.01


    # # knn_train_acc, knn_test_acc, knn_confusion_matrix = knn(train_bow, train_data.target, test_bow, test_data.target, 1)
    # # gnb_train_acc, gnb_test_acc, gnb_confusion_matrix = gnb(train_bow.todense(), train_data.target, test_bow.todense(), test_data.target)
    svm_train_acc, svm_test_acc, svm_confusion_matrix = svm(train_bow, train_data.target, test_bow, test_data.target, hyperparameter_index=1)
    mnb_train_acc, mnb_test_acc, mnb_confusion_matrix = mnb(train_bow.todense(), train_data.target, test_bow.todense(), test_data.target, hyper_parameter_info=0.01)
    rf_train_acc, rf_test_acc, rf_confusion_matrix = random_forest(train_bow, train_data.target, test_bow, test_data.target, hyper=500)
    bnb_train_acc, bnb_test_acc, bnb_confusion_matrix = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)


    # # np.fill_diagonal(knn_confusion_matrix, 0)
    # # np.fill_diagonal(gnb_confusion_matrix, 0)
    np.fill_diagonal(svm_confusion_matrix, 0)
    np.fill_diagonal(mnb_confusion_matrix, 0)
    np.fill_diagonal(rf_confusion_matrix, 0)
    np.fill_diagonal(bnb_confusion_matrix, 0)

    # # most_confused_knn = np.unravel_index(knn_confusion_matrix.argmax(), knn_confusion_matrix.shape)
    # # most_confused_gnb = np.unravel_index(gnb_confusion_matrix.argmax(), gnb_confusion_matrix.shape)
    most_confused_svm = np.unravel_index(svm_confusion_matrix.argmax(), svm_confusion_matrix.shape)
    most_confused_mnb = np.unravel_index(mnb_confusion_matrix.argmax(), mnb_confusion_matrix.shape)
    most_confused_rf = np.unravel_index(rf_confusion_matrix.argmax(), rf_confusion_matrix.shape)
    most_confused_bnb = np.unravel_index(bnb_confusion_matrix.argmax(), bnb_confusion_matrix.shape)

    # # print("For KNN, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
    #       # .format(knn_train_acc, knn_test_acc, most_confused_knn))
    #
    # # print("For gaussian naive bayes, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
    # #       .format(gnb_train_acc, gnb_test_acc, most_confused_gnb))
    #
    print("For SVM, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
          .format(svm_train_acc, svm_test_acc, most_confused_svm))

    print("For Multinomial Naive Bayes, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
          .format(mnb_train_acc, mnb_test_acc, most_confused_mnb))

    print("For Random Forest, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
          .format(rf_train_acc, rf_test_acc, most_confused_rf))

    print("For bernoulli naive bayes, train accuracy is: {}, test accuracy is:{}, most confused classes are {}."
          .format(bnb_train_acc, bnb_test_acc, most_confused_bnb))


    stop = datetime.now()
    print("run time is: {}".format(stop-start))




if __name__ == '__main__':
    main()