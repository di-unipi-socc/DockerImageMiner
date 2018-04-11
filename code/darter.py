import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import median_absolute_error


def get_knee_point_value(values):
    y = values
    x = np.arange(0, len(y))

    index = 0
    max_d = -float('infinity')

    for i in range(0, len(x)):
        c = closest_point_on_segment(a=[x[0], y[0]], b=[x[-1], y[-1]], p=[x[i], y[i]])
        d = np.sqrt((c[0] - x[i])**2 + (c[1] - y[i])**2)
        if d > max_d:
            max_d = d
            index = i

    return index


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    if u < 0:
        closest_point = a
    elif u > 1:
        closest_point = b
    else:
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        closest_point = [cp_x, cp_y]

    return closest_point


default_clf_param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                          'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
                          'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
                          'class_weight': [{0: 1 - x, 1: x} for x in np.arange(0.6, 1.0, 0.01)],
                          }

default_reg_param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 5, 10],
                          }


class DARTER:

    def __init__(self, clf_param_list=None, reg_param_list=None):
        self.clf_criterion = 'gini'
        self.reg_criterion = 'mae'
        self.clf_param_list = default_clf_param_list if clf_param_list is None else clf_param_list
        self.reg_param_list = default_reg_param_list if reg_param_list is None else reg_param_list

        self.clf = DecisionTreeClassifier(criterion=self.clf_criterion,
                                          max_depth=None, min_samples_split=2, min_samples_leaf=1)

        self.reg_s = DecisionTreeRegressor(criterion=self.reg_criterion,
                                           max_depth=None, min_samples_split=2, min_samples_leaf=1)

        self.reg_f = DecisionTreeRegressor(criterion=self.reg_criterion,
                                           max_depth=None, min_samples_split=2, min_samples_leaf=1)

    def _split_dataset(self, X, y, s):
        X_s = list()
        y_s = list()
        X_f = list()
        y_f = list()
        if y is not None:
            for x0, y0, c in zip(X, y, s):
                if c == 1:
                    X_s.append(x0)
                    y_s.append(y0)
                else:
                    X_f.append(x0)
                    y_f.append(y0)
        else:
            pos = 0
            for x0, c in zip(X, s):
                if c == 1:
                    X_s.append(x0)
                    y_s.append(pos)
                else:
                    X_f.append(x0)
                    y_f.append(pos)
                pos += 1

        X_s = np.array(X_s) #if len(X_s) > 1 else np.array([X_s])
        y_s = np.array(y_s) #if len(y_s) > 1 else np.array([y_s])
        X_f = np.array(X_f) #if len(X_f) > 1 else np.array([X_f])
        y_f = np.array(y_f) #if len(y_f) > 1 else np.array([y_f])
        return X_s, y_s, X_f, y_f

    def fit(self, X, y):

        y_train = y
        X_train = X

        # find threshold
        score_values = sorted(y_train, reverse=True)
        knee_index = get_knee_point_value(score_values)
        knee_index = 1 if knee_index == 0 else knee_index
        knee_index = len(score_values) - 2 if knee_index == len(score_values) - 1 else knee_index
        threshold = score_values[knee_index]
        self.success_threshold = threshold
        y_train_c = np.array([0 if y <= threshold else 1 for y in y_train])
        _, value_counts = np.unique(y_train_c, return_counts=True)
        nbr0 = value_counts[0]
        nbr1 = value_counts[1]

        if nbr1 <= 3:
            random_search_f = RandomizedSearchCV(self.reg_f, param_distributions=self.reg_param_list, n_iter=100,
                                                 refit=self.reg_criterion,
                                                 scoring=make_scorer(median_absolute_error))
            random_search_f.fit(X_train, y_train)
            self.reg_f = random_search_f.best_estimator_
            self.clf = None
            self.reg_s = None
        elif nbr0 <= 3:
            random_search_s = RandomizedSearchCV(self.reg_s, param_distributions=self.reg_param_list, n_iter=100,
                                                 refit=self.reg_criterion,
                                                 scoring=make_scorer(median_absolute_error))
            random_search_s.fit(X_train, y_train)
            self.reg_s = random_search_s.best_estimator_
            self.clf = None
            self.reg_f = None
        else:
            # train decision tree
            random_search_clf = RandomizedSearchCV(self.clf, param_distributions=self.clf_param_list, n_iter=1000,
                                                   scoring=make_scorer(f1_score, pos_label=1))
            random_search_clf.fit(X_train, y_train_c)
            self.clf = random_search_clf.best_estimator_

            # split dataset
            y_train_pred = self.clf.predict(X_train)
            values, value_counts = np.unique(y_train_pred, return_counts=True)
            if len(values) == 1 and values[0] == 0:
                nbr0 = value_counts[0]
                nbr1 = 0
            elif len(values) == 1 and values[0] == 1:
                nbr1 = value_counts[0]
                nbr0 = 0
            else:
                nbr0 = value_counts[0]
                nbr1 = value_counts[1]

            if nbr1 <= 3:
                random_search_f = RandomizedSearchCV(self.reg_f, param_distributions=self.reg_param_list, n_iter=100,
                                                     refit=self.reg_criterion,
                                                     scoring=make_scorer(median_absolute_error))
                random_search_f.fit(X_train, y_train)
                self.reg_f = random_search_f.best_estimator_
                self.clf = None
                self.reg_s = None
            elif nbr0 <= 3:
                random_search_s = RandomizedSearchCV(self.reg_s, param_distributions=self.reg_param_list, n_iter=100,
                                                     refit=self.reg_criterion,
                                                     scoring=make_scorer(median_absolute_error))
                random_search_s.fit(X_train, y_train)
                self.reg_s = random_search_s.best_estimator_
                self.clf = None
                self.reg_f = None
            else:
                X_train_s, y_train_s, X_train_f, y_train_f = self._split_dataset(X_train, y_train, y_train_pred)

                # train regressors
                random_search_s = RandomizedSearchCV(self.reg_s, param_distributions=self.reg_param_list, n_iter=100,
                                                     refit=self.reg_criterion,
                                                     scoring=make_scorer(median_absolute_error))
                random_search_s.fit(X_train_s, y_train_s)
                self.reg_s = random_search_s.best_estimator_

                random_search_f = RandomizedSearchCV(self.reg_f, param_distributions=self.reg_param_list, n_iter=100,
                                                     refit=self.reg_criterion,
                                                     scoring=make_scorer(median_absolute_error))
                random_search_f.fit(X_train_f, y_train_f)
                self.reg_f = random_search_f.best_estimator_

    def predict(self, X):
        X_test = X
        if self.clf is not None:
            y_pred_s, y_pred_f, y_pos_s, y_pos_f = self.predict_reg(X_test)

            y_pred = [0] * len(X_test)
            for pos, val in zip(y_pos_s, y_pred_s):
                y_pred[pos] = val
            for pos, val in zip(y_pos_f, y_pred_f):
                y_pred[pos] = val

            if len(X_test) > 1:
                return np.array(y_pred)
            else:
                return y_pred[0]
        else:
            if self.reg_f is None:
                return self.reg_s.predict(X_test)
            else:
                return self.reg_f.predict(X_test)

    def predict_clf(self, X):
        X_test = X
        if len(X_test) > 1:
            return self.clf.predict(X_test)
        else:
            return self.clf.predict(X_test.reshape(1, -1))

    def predict_reg(self, X):
        X_test = X
        y_pred_c = self.predict_clf(X_test)
        X_test_s, y_pos_s, X_test_f, y_pos_f = self._split_dataset(X_test, None, y_pred_c)

        if len(X_test_s) > 0:
            y_pred_s = self.reg_s.predict(X_test_s) if len(X_test_s) > 1 else self.reg_s.predict(X_test_s.reshape(1, -1))
        else:
            y_pred_s = np.array([])

        if len(X_test_f) > 0:
            y_pred_f = self.reg_f.predict(X_test_f) if len(X_test_f) > 1 else self.reg_f.predict(X_test_f.reshape(1, -1))
        else:
            y_pred_f = np.array([])

        return y_pred_s, y_pred_f, y_pos_s, y_pos_f

