from darter import *
from docker_manager import *

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def get_paths(clf):
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    path = list()
    paths = list()  # defaultdict(list)
    tree_paths(0, path, paths, clf, children_left, children_right)
    return paths


def tree_paths(node, path, paths, clf, children_left, children_right):
    if node == -1:
        return
    path.append(node)
    if children_left[node] == children_right[node]:
        pred = np.argmax(clf.tree_.value[node])
        paths.append([list(path), pred])
        # paths[pred].append(list(path))
    tree_paths(children_left[node], path, paths, clf, children_left, children_right)
    tree_paths(children_right[node], path, paths, clf, children_left, children_right)
    path.pop()


def get_paths_reg(reg):
    children_left = reg.tree_.children_left
    children_right = reg.tree_.children_right
    path = list()
    paths = list()  # defaultdict(list)
    tree_paths_reg(0, path, paths, reg, children_left, children_right)
    return paths


def tree_paths_reg(node, path, paths, reg, children_left, children_right):
    if node == -1:
        return
    path.append(node)
    if children_left[node] == children_right[node]:
        pred = reg.tree_.value[node][0][0]
        paths.append([list(path), pred])
        # paths[pred].append(list(path))
    tree_paths_reg(children_left[node], path, paths, reg, children_left, children_right)
    tree_paths_reg(children_right[node], path, paths, reg, children_left, children_right)
    path.pop()


def path_diff(path1, path2):

    p1 = [(path1[i], path1[i + 1]) for i in range(len(path1)-1)]
    p2 = [(path2[i], path2[i + 1]) for i in range(len(path2)-1)]

    i = 0
    while i < len(p1) and i < len(p2) and p1[i] == p2[i]:
        i += 1

    return len(p2[i:])


def get_counterfactual(sample_id, paths, model, node_indicator, mtype='clf'):
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    if mtype == 'clf':
        predx = np.argmax(model.tree_.value[node_index[-1]])
    if mtype == 'reg':
        predx = model.tree_.value[node_index[-1]][0][0]

    path_dist = list()
    for path, outcome in paths:
        if (mtype == 'clf' and outcome != predx) or (mtype == 'reg' and outcome > predx):
            dist = path_diff(node_index, path)
            path_dist.append([path, dist])

    if len(path_dist) > 0:
        return sorted(path_dist, key=lambda x: x[1])[0][0]
    else:
        return []


def impr_val(val):
    impr = 1
    while impr > val:
        impr /= 10
    return impr


def improve_image(X2E, sample_id, node_indicator, feature, threshold, cpath):
    path = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

    p1 = [(path[z], path[z + 1]) for z in range(len(path) - 1)]
    p2 = [(cpath[z], cpath[z + 1]) for z in range(len(cpath) - 1)]

    z = 0
    while z < len(p1) and z < len(p2) and p1[z] == p2[z]:
        z += 1

    improvement = list()
    while z < len(p2):
        node_id = p2[z][0]
        fpos = feature[node_id]
        if X2E[sample_id, feature[node_id]] <= threshold[z]:
            val = threshold[node_id] + impr_val(threshold[node_id])
        else:
            val = threshold[node_id]
        improvement.append((fpos, val))
        # fname = les[5][feature[node_id]]
        # thr_sign = '>' if (X2E[sample_id, feature[node_id]] <= threshold[node_id]) else '<='
        # print(fname, thr_sign, threshold[node_id], X2E[sample_id, feature[node_id]])
        z += 1

    # print(X2E[i])
    # print(improvement, '<<<<<<')
    improved = np.copy(X2E[sample_id])
    for fpos, val in improvement:
        improved[fpos] = val

    return improved, len(improvement)


def main():
    path = '../datasets/'
    path_exp = '../experiments/'
    target = 'pulls'

    officials = json.load(open('../officials_images.json', 'r', encoding='utf8'))['officials']

    repo_name_list = load_images(path)
    repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)

    repo_list = repo_list_latest

    repo_list = {k: v for k, v in repo_list.items() if k not in officials}

    X, y, les = dict2vector(repo_list, target=target, use_software_version=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

    darter_est = DARTER()
    darter_est.fit(X_train, y_train)
    y_pred = darter_est.predict_reg(X_test)
    y_pred_c = darter_est.predict_clf(X_test)

    model_f = darter_est.reg_f
    X2E = np.array([x for x, y in zip(X_test, y_pred_c) if y == 0])

    node_indicator = model_f.decision_path(X2E)
    paths = get_paths_reg(model_f)
    feature = model_f.tree_.feature
    threshold = model_f.tree_.threshold

    cpath_set = set()
    cfeatures2change = defaultdict(int)
    cfeatures2change_thr = defaultdict(int)
    cfeatures_importance = defaultdict(int)
    cfeatures_thr_importance = defaultdict(int)

    for i in range(len(X2E)):
        cpath = get_counterfactual(i, paths, model_f, node_indicator, mtype='reg')
        if len(cpath) == 0:
            continue
        cpath_set.add(tuple(cpath))
        path = node_indicator.indices[node_indicator.indptr[i]: node_indicator.indptr[i + 1]]
        cf_set = set()
        for node_id in cpath[:-1]:
            fname = les[5][feature[node_id]]
            cf_set.add(fname)
            thr_sign = '>' if (X2E[i, feature[node_id]] <= threshold[node_id]) else '<='
            cfeatures_thr_importance[(fname, thr_sign, threshold[node_id])] += 1

        for fname in cf_set:
            cfeatures_importance[fname] += 1

        p1 = [(path[z], path[z + 1]) for z in range(len(path) - 1)]
        p2 = [(cpath[z], cpath[z + 1]) for z in range(len(cpath) - 1)]
        z = 0
        while z < len(p1) and z < len(p2) and p1[z] == p2[z]:
            z += 1

        node_id = p2[z][0]
        fname = les[5][feature[node_id]]
        cfeatures2change[fname] += 1
        thr_sign = '>' if (X2E[i, feature[node_id]] <= threshold[node_id]) else '<='
        cfeatures2change_thr[(fname, thr_sign, threshold[node_id])] += 1

    for f in sorted(cfeatures_thr_importance, key=cfeatures_thr_importance.get, reverse=True)[:10]:
        print(f, cfeatures_thr_importance[f] / len(X2E))
    print('')

    for f in sorted(cfeatures_importance, key=cfeatures_importance.get, reverse=True)[:10]:
        print(f, cfeatures_importance[f] / len(X2E))
    print('')

    for f in sorted(cfeatures2change, key=cfeatures2change.get, reverse=True)[:10]:
        print(f, cfeatures2change[f] / len(X2E))
    print('')

    for f in sorted(cfeatures2change_thr, key=cfeatures2change_thr.get, reverse=True)[:10]:
        print(f, cfeatures2change_thr[f] / len(X2E))
    print('')


if __name__ == "__main__":
    main()

