import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, make_scorer , roc_curve
import pandas as pd
from metrics import eval
from prettytable import PrettyTable



def print_roc_curve(cf_lr, cf_xgb, cf_rf, X_test, y_test):
    fig = plt.figure()
    prob_lr = cf_lr.predict_proba(X_test)
    tpr_lr, fpr_lr, th = roc_curve(y_test, prob_lr[:, 1])
    prob_xgb = cf_xgb.predict_proba(X_test)
    tpr_xgb, fpr_xgb, th = roc_curve(y_test, prob_xgb[:, 1])
    prob_rf = cf_rf.predict_proba(X_test)
    tpr_rf, fpr_rf, th = roc_curve(y_test, prob_rf[:, 1])
    lr_plot, = plt.plot(tpr_lr, fpr_lr, 'b')
    xgb_plot, = plt.plot(tpr_xgb, fpr_xgb, 'g')
    rf_plot, = plt.plot(tpr_rf, fpr_rf, 'r')  ## add labels
    plt.legend((lr_plot, xgb_plot, rf_plot), ('lr', 'xgb', 'rf'))
    plt.title('roc curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.grid()
    #plt.savefig(output + "roc_curve.png", dpi=400)
    plt.show()
    
def print_violinplot(cf_lr, cf_xgb, cf_rf, X_test, y_test):
    prob_lr = cf_lr.predict_proba(X_test)
    prob_xgb = cf_xgb.predict_proba(X_test)
    prob_rf = cf_rf.predict_proba(X_test)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
    labels1 = []
    labels2 = []
    labels3 = []
    pos = np.ones((1, 1))
    all_pos = [1, 2, 3, 4]

    def add_label(violin, label, labels_i):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels_i.append((mpatches.Patch(color=color), label))

    add_label(ax1.violinplot(prob_lr[y_test == 1, 1], pos), "T1D/T1D", labels1)
    add_label(ax1.violinplot(prob_lr[y_test == 0, 1], pos * 2), "T1D/NO_T1D", labels1)
    
    ax1.set_title('lr')
    ax1.set_ylabel('probability')
    add_label(ax2.violinplot(prob_xgb[y_test == 1, 1], pos), "T1D/T1D", labels2)
    add_label(ax2.violinplot(prob_xgb[y_test == 0, 1], pos * 2), "T1D/NO_T1D", labels2)
    ax2.set_title('xgb')
    add_label(ax3.violinplot(prob_rf[y_test == 1, 1], pos), "T1D/T1D", labels3)
    add_label(ax3.violinplot(prob_rf[y_test == 0, 1], pos * 2), "T1D/NO_T1D", labels3)
    ax3.legend(*zip(*labels3), loc='center left', bbox_to_anchor=(1, 0.5), title="class_prob/right_class")
    ax3.set_title('rf')
    plt.show()
    
def print_hyperparameters_heatmap(cf_lr, cf_xgb, cf_rf, X_test, y_test, hyperparameters):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
    test_score_lr = np.expand_dims(cf_lr.cv_results_['mean_test_score'], 1)
    c_hyperparam = hyperparameters['LR']['C']
    im = ax1.imshow(test_score_lr, cmap='jet')
    ax1.set_yticks(np.arange(len(c_hyperparam)))
    ax1.set_yticklabels(c_hyperparam)
    ax1.set_ylabel('regulation term')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(c_hyperparam)):
        text = ax1.text(0, i, format(test_score_lr[i, 0], '.2f'), ha="center", va="center", color="w")
    ax1.set_title("lr heatmap")

    n_estimators_hyperparam = hyperparameters['XGB']['n_estimators']
    max_depth_hyperparam = hyperparameters['XGB']['max_depth']
    test_score_xgb = cf_xgb.cv_results_['mean_test_score'].reshape(len(max_depth_hyperparam),
                                                                   len(n_estimators_hyperparam))
    im = ax2.imshow(test_score_xgb, cmap='jet')
    ax2.set_yticks(np.arange(len(max_depth_hyperparam)))
    ax2.set_xticks(np.arange(len(n_estimators_hyperparam)))
    ax2.set_yticklabels(max_depth_hyperparam)
    ax2.set_xticklabels(n_estimators_hyperparam)
    ax2.set_ylabel('max depth')
    ax2.set_xlabel('num of estimators')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(max_depth_hyperparam)):
        for j in range(len(n_estimators_hyperparam)):
            text = ax2.text(j, i, format(test_score_xgb[i, j], '.2f'), ha="center", va="center", color="w")
    ax2.set_title("xgb heatmap")

    n_estimators_hyperparam = hyperparameters['RF']['n_estimators']
    max_depth_hyperparam = hyperparameters['RF']['max_depth']
    test_score_rf = cf_rf.cv_results_['mean_test_score'].reshape(len(max_depth_hyperparam),
                                                                 len(n_estimators_hyperparam))
    im = ax3.imshow(test_score_rf, cmap='jet')
    ax3.set_yticks(np.arange(len(max_depth_hyperparam)))
    ax3.set_xticks(np.arange(len(n_estimators_hyperparam)))
    ax3.set_yticklabels(max_depth_hyperparam)
    ax3.set_xticklabels(n_estimators_hyperparam)
    ax3.set_ylabel('max depth')
    ax3.set_xlabel('num of estimators')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(max_depth_hyperparam)):
        for j in range(len(n_estimators_hyperparam)):
            text = ax3.text(j, i, format(test_score_rf[i, j], '.2f'), ha="center", va="center", color="w")
    ax3.set_title("rf heatmap")
    fig.tight_layout()
    #plt.savefig(output + "heatmap.png", dpi=400)
    plt.show()

def print_result_tables(cf_lr, cf_xgb, cf_rf, X_test, y_test,metric_results,scores):
    if metric_results:
        metric_result = PrettyTable()
        result_cf_lr = eval(cf_lr, X_test, y_test)
        result_cf_xgb = eval(cf_xgb, X_test, y_test)
        result_cf_rf = eval(cf_rf, X_test, y_test)
        column_names = ["Metric", "LR", "XGB", "RF"]
        metric_result.add_column(column_names[0], ["Accuracy", "F1-Score", "Sensitivity",
                                   "Specificity", "PPV", "NPV", "AUROC"])
        metric_result.add_column(column_names[1], list(map('{:.2f}'.format,np.asarray(result_cf_lr))) )
        metric_result.add_column(column_names[2], list(map('{:.2f}'.format,np.asarray(result_cf_xgb))) )
        metric_result.add_column(column_names[3], list(map('{:.2f}'.format,np.asarray(result_cf_rf))) )


        print(metric_result)
    if scores:
        scores = PrettyTable()
        best_i_lr = cf_lr.best_index_
        mean_train_lr = format(cf_lr.cv_results_['mean_train_score'][best_i_lr], '.3f')
        mean_val_lr = format(cf_lr.cv_results_['mean_test_score'][best_i_lr], '.3f')
        std_train_lr = format(cf_lr.cv_results_['std_train_score'][best_i_lr],'.4f')
        std_val_lr = format(cf_lr.cv_results_['std_test_score'][best_i_lr],'.4f')
        best_i_xgb = cf_xgb.best_index_
        mean_train_xgb = format(cf_xgb.cv_results_['mean_train_score'][best_i_xgb], '.3f')
        mean_val_xgb = format(cf_xgb.cv_results_['mean_test_score'][best_i_xgb], '.3f')
        std_train_xgb = format(cf_xgb.cv_results_['std_train_score'][best_i_xgb],'.4f')
        std_val_xgb = format(cf_xgb.cv_results_['std_test_score'][best_i_xgb],'.4f')
        best_i_rf = cf_rf.best_index_
        mean_train_rf = format(cf_rf.cv_results_['mean_train_score'][best_i_rf], '.3f')
        mean_val_rf = format(cf_rf.cv_results_['mean_test_score'][best_i_rf], '.3f')
        std_train_rf = format(cf_rf.cv_results_['std_train_score'][best_i_rf],'.4f')
        std_val_rf = format(cf_rf.cv_results_['std_test_score'][best_i_rf],'.4f')

        scores.field_names = ["Classifier","Mean train score", "Std train score", "Mean validation score", "Std validation score" ,"test    score"]

        scores.add_row(["LR", mean_train_lr, std_train_lr, mean_val_lr,std_val_lr, format(result_cf_lr[0],'.2f')])
        scores.add_row(["XGB", mean_train_xgb, std_train_xgb, mean_val_xgb,std_val_xgb, format(result_cf_xgb[0],'.2f')])
        scores.add_row(["RF", mean_train_rf, std_train_rf, mean_val_rf,std_val_rf, format(result_cf_rf[0],'.2f')])

        print(scores)