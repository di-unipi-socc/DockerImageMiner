import datetime
import warnings

from darter import *
from docker_manager import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score, r2_score

warnings.filterwarnings("ignore")

models = {
    'knn': KNeighborsRegressor,
    'darter': DARTER,
    'dtreg': DecisionTreeRegressor,
    'linreg': LinearRegression,
    'lasso': Lasso,
    'ridge': Ridge,
}

filename_stability = 'stability_%s.csv'
header_stability = 'iter,model,dataset,mean_absolute_error,mean_squared_error,median_absolute_error,explained_variance_score,r2_score,mean_squared_log_error,fit_time,predict_time\n'

#
# def mean_relative_squared_error(y_real, y_pred):
#     rse_list = list()
#     for yr, yp in zip(y_real, y_pred):
#         rse = (yp/yr - 1)**2
#         if rse != np.inf:
#             rse_list.append(rse)
#     return float(np.mean(rse_list))


def run_experiment(iter, model_name, X_train, X_test, y_train, y_test, target, path_exp, dataset):

    model = models[model_name]()

    start_fit_time = datetime.datetime.now()
    try:
        model.fit(X_train, y_train)
    except Exception:
        return
    end_fit_time = datetime.datetime.now()

    y_pred = model.predict(X_test)
    end_predict_time = datetime.datetime.now()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae2 = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    r2s = r2_score(y_test, y_pred)
    y_pred2 = np.array([y0 if y0 > 0.0 else max(y_pred) for y0 in y_pred])
    msle = mean_squared_log_error(y_test, y_pred2)
    fit_time = (end_fit_time - start_fit_time).total_seconds()
    predict_time = (end_predict_time - end_fit_time).total_seconds()

    file_stability = open(path_exp + (filename_stability % target), 'a')
    file_stability.write('%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (
        iter, model_name, dataset, mae, mse, mae2, evs, r2s, msle, fit_time, predict_time))
    file_stability.close()


def main():
    path = '../datasets/'
    path_exp = '../experiments/'
    target = 'stars'

    file_stability = open(path_exp + (filename_stability % target), 'w')
    file_stability.write(header_stability)
    file_stability.close()

    nbr_iter = 10

    docker_files = sorted([filename for filename in listdir(path) if isfile(join(path, filename))])

    for dataset in docker_files:
        if not dataset.endswith('.json'):
            continue
        print('Dataset: %s' % dataset)
        repo_name_list = load_images_filename(path, dataset)
        repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)
        repo_list = repo_list_latest
        X, y, les = dict2vector(repo_list, target=target, use_software_version=False)
        for iter in range(0, nbr_iter):
            print('\tIteration %d' % iter)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=iter)
            for model_name in models:
                print('\t\tModel: %s' % model_name)
                run_experiment(iter, model_name, X_train, X_test, y_train, y_test, target, path_exp, dataset)


if __name__ == "__main__":
    main()
