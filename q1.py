from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    i: index

    for i in range(feature_count):
        axe = plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter(X[:, i], y, c='green', marker='.')
        axe.set_xlabel(features[i])
        axe.set_ylabel('target')
        
    plt.tight_layout()
    plt.show()


def fit_regression(X,y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    # raise NotImplementedError()
    biased_x = np.array([[1] + X[i] for i in range(len(X))])
    np_y = np.array(y)

    a = np.dot(np.transpose(biased_x), biased_x) # this is X^TX
    b = np.dot(np.transpose(biased_x), np_y)     # this is X^Ty
    w = np.linalg.solve(a, b)    # solve the equation

    return w

def report(X, y):
    # report the informations of the data
    print("number of target is {} ".format(y.shape[0]))
    print("The dimension of data is {} (row) * {} (colum) ".format(X.shape[0], X.shape[1]))

def divide_data(X, y):
    # divide data into training set and testing set.
    amount = len(X)
    training = np.random.choice(amount, amount * 4 // 5, replace=False)
    test = [n for n in range(amount) if n not in training]

    training_set = ([X[i] for i in training], [y[j] for j in training])
    test_set = ([X[i] for i in test], [y[j] for j in test])

    return training_set, test_set

def test_module(w, X, y):
    # calculate the MSE of the regression model
    biased_x = np.array([[1] + X[i] for i in range(len(X))])
    np_y = np.array(y)

    predict_y = np.dot(biased_x, w)
    MSE = np.divide(np.sum(np.power(np.subtract(np_y, predict_y), 2)), len(biased_x))
    print("MSE of this linear regression module is {}.".format(MSE))


def get_features_weight(w, features):
    # print the corresponding weight of features
    print("----------------------------")
    for i in range(len(features)):
        print("The weight of feature {} is {}".format(features[i], w[i]))

def main():
    # Load the data
    X, y, features = load_data()

    # Report the informations of data
    print("Features: {}".format(features))
    report(X, y)

    # Visualize the features
    # visualize(X, y, features)
    
    #TODO: Split data into train and test
    (training_x, training_y), (test_x, test_y) = divide_data(X, y)
    #check the size of traning data and testing data are correct.
    print("the size of training_x: {}, training_y: {}.".format(len(training_x), len(training_y)))
    print("the size of test_x: {}, test_y: {}.".format(len(test_x), len(test_y)))

    # Fit regression model
    w = fit_regression(training_x, training_y)

    # report the weight of features
    get_features_weight(w, features)

    # Compute fitted values, MSE, etc.
    test_module(w, test_x, test_y)


if __name__ == "__main__":
    main()

