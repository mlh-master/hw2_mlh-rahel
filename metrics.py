from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve
import pandas as pd
import numpy as np
import sys


def print_met(accuracy, fbeta, se, sp, PPV, NPV, AUROC, beta):
    """ This function prints the different metrics reeived as input.
    :param accuracy:            The accuracy measure.
    :param fbeta:               The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :param AUROC:               The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :param sensitivity:         The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param specificity:         The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param PPV:                 The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Precision_and_recall)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """
    print("Accuracy: " + str(accuracy))
    print("F" + str(beta) + "-Score: " + str(fbeta))
    print("Sensitivity: " + str(se))
    print("Specificity: " + str(sp))
    print("PPV: " + str(PPV))
    print("NPV: " + str(NPV))
    print("AUROC: " + str(AUROC))


def model_metrics(X, y, y_hat, print_metrics=True, beta=1):
    """ This function returns different statistical binary metrics based on the data (output score/probabilities),
        the predicted and the actual labels. Function established for binary classification only.
    :param X:                   The output score/probabilities of the algorithm.
    :param y:                   The actual labels of the examples.
    :param y_hat:               The predicted labels of the examples.
    :param beta:                Index for the F-beta measure.
    :param print_metrics:       Boolean value to print or not the mtrics. Default is True
    :returns accuracy:          The accuracy measure.
    :returns fbeta:             The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :returns AUROC:             The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :returns sensitivity:       The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns specificity:       The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns PPV:               The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """
    AUROC = roc_auc_score(y, X)
    accuracy = accuracy_score(y, y_hat)
    TN, FP, FN, TP = confusion_matrix(y, y_hat).ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if np.isnan(precision):
        precision = sys.float_info.epsilon
    if np.isnan(recall):
        recall = sys.float_info.epsilon

    sensitivity = recall
    specificity = TN / (TN + FP)
    NPV = TN / (TN + FN)
    PPV = precision
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    if np.isnan(fbeta):
        fbeta = sys.float_info.epsilon
    if print_metrics:
        print_met(accuracy, fbeta, sensitivity, specificity, PPV, NPV, AUROC, beta)
        print(confusion_matrix(y, y_hat))
    return accuracy, fbeta, sensitivity, specificity, PPV, NPV, AUROC


def eval(clf, X_new, y_new, sign=1, print_metrics=False, threshold=None, beta=1):

    """ This function evaluates the performance statistics of a given classifier and returns them.
        The classifier is assumed to implement the interface of sklearn classifiers (object which
        should have the following methods: predict, predict_proba).

    :param clf:                 The input classifier already trained.
    :param X_new:               The raw data on which the classifier has been trained (numpy array with dimensions (n_samples, n_features).
    :param y_new:               The actual labels of the samples.
    :param sign:                The direction of the decision function ( '<=' or '>=' for weak classifiers).
    :param beta:                Index for the F-beta measure computation.
    :param print_metrics:       Boolean value to print or not the mtrics. Default is True
    :param threshold:           Threshold on the decision scores (output of clf.predict_proba) for the positive class. If None, set at 0.5
    :returns accuracy:          The accuracy measure.
    :returns fbeta:             The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :returns AUROC:             The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :returns sensitivity:       The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns specificity:       The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns PPV:               The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """

    if threshold is None:
        predicted = clf.predict(X_new)
    else:
        predicted = clf.predict_proba(X_new)[:, 1] > threshold
    pred_score = clf.predict_proba(X_new)[:, -1]
    return model_metrics(sign * pred_score, y_new, predicted, print_metrics, beta)


