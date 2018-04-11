import math
import warnings
import numpy as np
import scipy.stats as st

from docker_manager import *

import pandas as pd

warnings.filterwarnings("ignore")

# Distributions to check
DISTRIBUTIONS = [st.uniform, st.dweibull, st.exponweib, st.expon, st.exponnorm, st.gamma, st.beta, st.alpha,
                 st.chi, st.chi2, st.laplace, st.lognorm, st.norm, st.powerlaw]


def freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = 2.0 * iqr / n** (1.0 / 3.0)
    k = math.ceil((np.max(x) - np.min(x)) / h)
    return k


def struges(x):
    n = len(x)
    k = math.ceil(np.log2(n)) + 1
    return k


def estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # print 'aaa'
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                # print distribution.name, sse
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params


def get_distr_values(x, size=1000):
    nbr_bins = int(np.round(estimate_nbr_bins(x)))
    name, params = best_fit_distribution(x, nbr_bins)
    dist = getattr(st, name)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    distr_values = np.linspace(start, end, size)

    return distr_values


def get_prob_distributions(repo_list, n=1000, use_software_version=False):
    byte2gigabyte = 1073741824

    size = list()
    csize = list()
    nlayers = list()
    nsoftwares = list()
    distro_count = defaultdict(int)
    software_count = defaultdict(int)

    for repo_name, repo in repo_list.items():
        size.append(repo['size'] / byte2gigabyte)
        csize.append(repo['complete_size'] / byte2gigabyte)
        nlayers.append(repo['nbr_layers'])
        nsoftwares.append(repo['nbr_softwares'])
        distro_count[repo['distro']] += 1
        for s in repo['softwares']:
            slabel = '%s%s' % (s[0], s[1]) if use_software_version else '%s' % (s[0])
            software_count[slabel] += 1

    size = get_distr_values(size, n).tolist()
    csize = get_distr_values(csize, n).tolist()
    nlayers = get_distr_values(nlayers, n).tolist()
    nsoftwares = get_distr_values(nsoftwares, n).tolist()

    d = sorted(distro_count.keys())
    distro = np.random.choice(d, n, p=[distro_count[x] / len(repo_list) for x in d])

    s = sorted(software_count.keys())
    tot_s = np.sum(list(software_count.values()))

    return size, csize, nlayers, nsoftwares, distro, s, [software_count[x] / tot_s for x in s]


def get_uniform_prob_distributions(repo_list, n=1000, use_software_version=False):
    byte2gigabyte = 1073741824

    size = list()
    csize = list()
    nlayers = list()
    nsoftwares = list()
    distro_count = defaultdict(int)
    software_count = defaultdict(int)

    for repo_name, repo in repo_list.items():
        size.append(repo['size'] / byte2gigabyte)
        csize.append(repo['complete_size'] / byte2gigabyte)
        nlayers.append(repo['nbr_layers'])
        nsoftwares.append(repo['nbr_softwares'])
        distro_count[repo['distro']] += 1
        for s in repo['softwares']:
            slabel = '%s%s' % (s[0], s[1]) if use_software_version else '%s' % (s[0])
            software_count[slabel] += 1

    size = np.random.normal(np.mean(size), np.std(size), n)
    csize = np.random.normal(np.mean(csize), np.std(csize), n)
    nlayers = np.random.normal(np.mean(nlayers), np.std(nlayers), n)
    nsoftwares = np.random.normal(np.mean(nsoftwares), np.std(nsoftwares), n)

    d = sorted(distro_count.keys())
    distro = np.random.choice(d, n)

    s = sorted(software_count.keys())
    ssoftwares = np.random.choice(s, n)

    return size, csize, nlayers, nsoftwares, distro, ssoftwares


def generate_random_images(size, csize, nlayers, nsoftwares, distro, softwares, softwarep, nbr_images=1000):
    images = dict()
    for i in range(0, nbr_images):
        nbr_softwares = int(np.round(np.random.choice(nsoftwares, 1)[0]))

        csize_val = np.random.choice(csize, 1)[0]
        size_val = np.random.choice(size, 1)[0]
        while csize_val <= size_val:
            size_val = np.random.choice(size, 1)[0]

        repo = {
            'size': size_val,
            'complete_size': csize_val,
            'nbr_layers': int(np.round(np.random.choice(nlayers, 1)[0])),
            'nbr_softwares': nbr_softwares,
            'distro': np.random.choice(distro, 1)[0],
            'softwares': np.random.choice(softwares, nbr_softwares, p=softwarep, replace=False).tolist(),
        }
        images[len(images)] = repo
    return images


def generate_uniform_random_images(size, csize, nlayers, nsoftwares, distro, softwares, nbr_images=1000):
    images = dict()
    for i in range(0, nbr_images):
        nbr_softwares = int(np.round(np.random.choice(nsoftwares, 1)[0]))

        csize_val = np.random.choice(csize, 1)[0]
        size_val = np.random.choice(size, 1)[0]
        while csize_val <= size_val:
            size_val = np.random.choice(size, 1)[0]

        repo = {
            'size': size_val,
            'complete_size': csize_val,
            'nbr_layers': int(np.round(np.random.choice(nlayers, 1)[0])),
            'nbr_softwares': nbr_softwares,
            'distro': np.random.choice(distro, 1)[0],
            'softwares': np.random.choice(softwares, nbr_softwares, replace=False).tolist(),
        }
        images[len(images)] = repo
    return images


def random_change_image(base, size, csize, nlayers, nsoftwares, distro, softwares, softwarep,
                        distro_labels, software_labels, p=0.2):
    improved = np.copy(base)
    nbr_changes = 0
    for i in range(0, 5):
        if np.random.random() > (1.0 - p):
            if i == 0:
                nbr_changes += 1
                improved[i] = np.random.choice(size, 1)[0]
            elif i == 1:
                nbr_changes += 1
                improved[i] = np.random.choice(csize, 1)[0]
            elif i == 2:
                nbr_changes += 1
                improved[i] = int(np.round(np.random.choice(nlayers, 1)[0]))
            elif i == 3:
                nbr_softwares = int(np.round(np.random.choice(nsoftwares, 1)[0]))
                if nbr_softwares < base[i]:
                    to_remove = base[i] - nbr_softwares
                    removed = 0
                    while removed < to_remove:
                        j = np.random.randint(len(distro_labels), len(base))
                        if improved[j] == 1:
                            improved[j] = 0
                            removed += 1
                            nbr_changes += 1
                elif nbr_softwares > base[i]:
                    to_add = nbr_softwares - base[i]
                    added = 0
                    while added < to_add:
                        j = software_labels[np.random.choice(softwares, 1, p=softwarep).tolist()[0]]
                        if improved[j] == 0:
                            improved[j] = 1
                            added += 1
                            nbr_changes += 1
                improved[i] = nbr_softwares
            elif i == 4:
                nbr_changes += 1
                for j in range(4, len(distro_labels)):
                    improved[j] = 0
                improved[i] = distro_labels[np.random.choice(distro, 1)[0]]

    return improved, nbr_changes


def main():
    path = '../datasets/'
    path_exp = '../experiments/'
    target = 'pulls'

    repo_name_list = load_images(path)
    repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)

    repo_list = repo_list_latest
    X, y, les = dict2vector(repo_list, target=target, use_software_version=False)
    distro_labels = les[0]
    software_labels = les[2]

    repo_list = repo_list_latest
    # size, csize, nlayers, nsoftwares, distro, softwares, softwarep = get_prob_distributions(repo_list, n=1000)
    #
    # random_images = generate_random_images(size, csize, nlayers, nsoftwares, distro, softwares, softwarep, nbr_images=10)
    # print(random_images[0])
    # print('--------')
    # print(X[0])
    # print(random_change_image(X[0], size, csize, nlayers, nsoftwares, distro, softwares, softwarep,
    #                           distro_labels, software_labels, p=0.2))

    size, csize, nlayers, nsoftwares, distro, softwares = get_uniform_prob_distributions(repo_list, n=1000)
    random_images = generate_uniform_random_images(size, csize, nlayers, nsoftwares, distro, softwares,
                                                   nbr_images=10)
    print(random_images[0])




if __name__ == "__main__":
    main()