from dim import *
from darter import *
from docker_manager import *
from docker_generator import *

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


header_improvement = 'image_id,method,target,estimation,improvement,delta,nbr_changes\n'
filename_improvement = 'improvement_%s.csv'


def main():
    path = '../datasets/'
    path_exp = '../experiments/'
    target = 'pulls'

    repo_name_list = load_images(path)
    repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)

    repo_list = repo_list_latest
    X, y, les = dict2vector(repo_list, target=target, use_software_version=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
    size, csize, nlayers, nsoftwares, distro, softwares, softwarep = get_prob_distributions(repo_list, n=1000)

    distro_labels = les[0]
    software_labels = les[2]

    darter_est = DARTER()
    darter_est.fit(X_train, y_train)

    y_pred_c = darter_est.predict_clf(X_test)

    model_f = darter_est.reg_f
    X2E = np.array([x for x, y in zip(X_test, y_pred_c) if y == 0])
    y2E = np.array([x for x, y in zip(y_test, y_pred_c) if y == 0])

    node_indicator = model_f.decision_path(X2E)
    paths = get_paths_reg(model_f)
    feature = model_f.tree_.feature
    threshold = model_f.tree_.threshold

    file_improvement = open(path_exp + (filename_improvement % target), 'w')
    file_improvement.write(header_improvement)

    for i in range(len(X2E)):
        y0 = darter_est.predict(np.array([X2E[i]]))
        cpath = get_counterfactual(i, paths, model_f, node_indicator, mtype='reg')

        if len(cpath) == 0:
            continue

        impr, nbr_changes = improve_image(X2E, i, node_indicator, feature, threshold, cpath)
        y = darter_est.predict(np.array([impr]))
        delta = y - y0
        file_improvement.write('%d,dim,%s,%s,%s,%s,%d\n' % (i, y2E[i], y0, y, delta, nbr_changes))

        for p in np.arange(0.1, 1.0, 0.1):
            impr, nbr_changes = random_change_image(X2E[i], size, csize, nlayers, nsoftwares, distro, softwares,
                                                softwarep, distro_labels, software_labels, p)
            y = darter_est.predict(np.array([impr]))
            delta = y - y0
            file_improvement.write('%d,rnd%s,%s,%s,%s,%s,%d\n' % (i, p, y2E[i], y0, y, delta, nbr_changes))

    file_improvement.close()


if __name__ == "__main__":
    main()

