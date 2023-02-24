# coding:utf-8
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, matthews_corrcoef, \
    confusion_matrix
import time
from itertools import product
from sklearn.feature_selection import VarianceThreshold

np.random.seed(17)
# svm_grid
parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-15, 3, 10, base=2), 'C': np.logspace(-5, 9, 8, base=2),
                  'class_weight': ['balanced']}
# nn_grid
parameters_nn = {"learning_rate": ["constant"],  # learning_rate:{"constant","invscaling","adaptive"} default constant
                 "max_iter": [10000],
                 "hidden_layer_sizes": [(50,), (100,), (200,), (300,), (500,)],
                 "alpha": 10.0 ** -np.arange(1, 7),
                 "activation": ["relu"],  # "identity","tanh","relu","logistic"
                 "solver": ["adam"]}  # "lbfgs","adam""sgd"
# knn_grid
parameters_knn = {"n_neighbors": range(3, 10, 2), "weights": ['distance', "uniform"]}
# rf_grid
parameters_rf = {"min_samples_split": range(3, 5),
                 "min_samples_leaf": range(3, 5),
                 "n_estimators": range(10, 121, 10),
                 "criterion": ["gini", "entropy"],
                 "class_weight": ["balanced_subsample", "balanced"]}
# nb_grid
parameters_nb = {}
# dt_grid
parameters_dt = {"criterion": ['entropy', 'gini'], "splitter": ['best', 'random'], "max_depth": range(5, 50, 5), }
# lr_grid
parameters_lr = {"solver": ["liblinear", "lbfgs", "newton-cg", "sag"], "C": np.arange(0.01, 10.01, 0.5)}
parameters_gbdt ={"n_estimators": range(10, 101, 10), "learning_rate":np.arange(1, 21,1)*0.1}
# et_grid
parameters_et = {"n_estimators": range(10, 101, 10), "max_depth": range(5, 50, 5), "criterion": ["gini", "entropy"]}
model_map = {"svm": SVC, "knn": KNeighborsClassifier, "nn": MLPClassifier, "rf": RandomForestClassifier,
             "nb": GaussianNB, "dt": DecisionTreeClassifier, "lr": LogisticRegression, "et": ExtraTreesClassifier,"gbdt": GradientBoostingClassifier}
fp_map = {"maccs": "MACCSFP", "fp": "FP", "subfp": "SubFP", "pubchemfp": "PubchemFP"}
parameter_grid = {"svm": parameters_svm, "knn": parameters_knn, "nn": parameters_nn, "rf": parameters_rf,
                  "nb": parameters_nb, "dt": parameters_dt, "lr": parameters_lr, "et": parameters_et,"gbdt": parameters_gbdt}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
sample_map = {"ros": RandomOverSampler(), "smote": SMOTE(), "cc": ClusterCentroids(), "rus": RandomUnderSampler(),
              "smote_enn": SMOTEENN(), "smote_tomek": SMOTETomek()}


# evaluation metrics
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, average="binary")


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")


def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

def f1(y_true,y_pred):
    return f1_score(y_true, y_pred)


def classifer_generator(tuned_parameters, method, train_x, train_y, n_jobs=10):
    """
    Return the best model and the parameters of the model'
    """
    if method == SVC:
        grid = GridSearchCV(method(probability=True, random_state=17), param_grid=tuned_parameters, scoring="accuracy",
                            cv=cv, n_jobs=n_jobs)
    elif method == KNeighborsClassifier:
        grid = GridSearchCV(method(random_state=17), param_grid=tuned_parameters, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    else:
        grid = GridSearchCV(method(random_state=17), param_grid=tuned_parameters, scoring="accuracy", cv=cv,
                            n_jobs=n_jobs)
    grid.fit(train_x, train_y)
    return grid.best_estimator_, grid.best_params_


# print(grid.best_params_)

def data_reader(filename):
    'Read fingerprint file'
    data = pd.read_csv(filename, header=None).values
    x = data[:, 1:]
    y = data[:, 0]
    return x, y


def cv_results(best_model, train_x, train_y):
    'Return the performance of the cross-validation'
    ACC = []
    SE = []
    SP = []
    MCC = []
    AUC = []
    F1 = []
    for train_index, test_index in cv.split(train_x, train_y):
        y_pred = []
        y_true = []
        y_scores = []
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        m = best_model.fit(x_train, y_train)
        y_true.extend(y_test)
        y_pred.extend(m.predict(x_test))
        y_scores.extend(m.predict_proba(x_test)[:, 1])
        ACC.append(accuracy(y_true, y_pred))
        SE.append(recall(y_true, y_pred))
        AUC.append(auc(y_true, y_scores))
        MCC.append(mcc(y_true, y_pred))
        SP.append(sp(y_true, y_pred))
        F1.append(f1(y_true,y_pred))
    return np.mean(ACC), np.std(ACC), np.mean(SE), np.std(SE), np.mean(AUC), np.std(AUC), np.mean(MCC), np.std(MCC), np.mean(SP), np.std(SP),np.mean(F1), np.std(F1)


def test_results(best_model, train_x, train_y, test_x, test_y):
    'Return the performance of the test validation'
    y_true = test_y
    model = best_model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_scores = best_model.predict_proba(test_x)[:, 1]
    return accuracy(y_true, y_pred), recall(y_true, y_pred), auc(y_true, y_scores), mcc(
        y_true, y_pred), sp(y_true, y_pred),f1(y_true,y_pred)


def search_best_model(training_data, method_name, test_data):
    'Return {"model": best_model, "cv": metrics_of_cv,"tv": metrics_of_test,"parameter": best_parameter", "method": method_name}'
    train_x = training_data[0]
    train_y = training_data[1]
    tuned_parameters = parameter_grid[method_name]
    method = model_map[method_name]
    cg = classifer_generator(tuned_parameters, method, train_x, train_y)
    best_model = cg[0]
    parameter = cg[1]
    cv_metrics = cv_results(best_model, train_x, train_y)
    result = {'model': best_model, 'cv': cv_metrics, 'method': method_name}
    test_X = test_data[0]
    test_y = test_data[1]
    test_metrics = test_results(best_model, train_x, train_y, test_X, test_y)
    result['tv'] = test_metrics
    return result, result['tv'], parameter


def main(model_list, fp_list, targets, fpsz_list, radius_list):
    model_names = product(model_list, fp_list, targets, fpsz_list, radius_list)
    metrics_file = open("results_cv_10_metrics.txt", "w")
    test_file = open("results_test_metrics.txt", "w")
    train_file = open("results_train_metrics.txt", "w")
    para_file = open("para.txt", "w")
    metrics_file.write("Target\tFingerprint\tFpsz\tMethod\tRadius\tACC\tstd\tSE\tstd\tAUC\tstd\tMCC\tstd\tSP\tstd\tF1\tstd\n")
    test_file.write("Target\tFingerprint\tFpsz\tMethod\tRadius\tACC\tSE\tAUC\tMCC\tSP\tF1\n")
    train_file.write("Target\tFingerprint\tFpsz\tMethod\tRadius\tACC\tSE\tAUC\tMCC\tSP\tF1\n")
    for method_name, fp_name, target, fpsz, radius in model_names:
        print(method_name, fp_name, target)
        training_data = data_reader(target + '_' + fp_name + '_' + fpsz + '_' + radius + '.csv')
        test_data = data_reader('test' + '_' + fp_name + '_' + fpsz + '_' + radius + '.csv')
        model_results = search_best_model(training_data, method_name, test_data)[0]
        test_results = search_best_model(training_data, method_name, test_data)[1]
        train_results = search_best_model(training_data, method_name, training_data)[1]
        para = search_best_model(training_data, method_name, test_data)[2]
        cv_res = [str(x) for x in model_results['cv']]
        test_res = [str(x) for x in test_results]
        train_res = [str(x) for x in train_results]
        metrics_file.write(
            '%s\t%s\t%s\t%s\t%s\t%s\n' % (target, fp_name, fpsz, radius, method_name,'\t'.join(cv_res)))
        test_file.write(
            '%s\t%s\t%s\t%s\t%s\t%s\n' % (target, fp_name, fpsz, radius, method_name,'\t'.join(test_res)))
        train_file.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (target, fp_name, fpsz, radius, method_name,'\t'.join(train_res)))
        para_file.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (target, fp_name, fpsz, radius, method_name, para))
    metrics_file.close()
    test_file.close()
    para_file.close()


if __name__ == '__main__':
    start = time.clock()
    model_list = ["rf", "svm","gbdt"]
    fp_list = ["Morgan"]
    targets = ['hAOX']
    fpsz_list = ["256", "512", "1024", "2048"]
    radius_list = ['1', '2', '3', '4', '5']
    main(model_list, fp_list, targets, fpsz_list, radius_list)

