import json
import numpy as np

from os import listdir
from os.path import isfile, join

from collections import defaultdict


def load_images_filename(path, filename):
    data = json.load(open(path + filename, 'r', encoding='utf8'))
    images = data['images']
    repo_name_list = defaultdict(list)

    for image in images:
        df_id = image['_id']
        last_updated = image['last_updated']
        last_scan = image['last_scan']
        name = image['name']
        repo_name = image.get('repo_name', '')

        tag = image.get('tag', '')
        stars = image['stars']
        pulls = image['pulls']
        distro = image['distro']
        size = image['size']

        if distro is None or image['inspect_info'] is None or image['softwares'] is None:
            continue

        os = image['inspect_info']['Os']
        architecture = image['inspect_info']['Architecture']
        sha_id = image['inspect_info']['Id']
        complete_size = image['inspect_info']['Size']
        virtual_size = image['inspect_info']['VirtualSize']
        layers = image['inspect_info']['RootFS']['Layers']

        softwares = [(s['software'], s['ver']) for s in image['softwares']]

        repo_name_record = {
            'df_id': df_id,
            'last_updated': last_updated,
            'last_scan': last_scan,
            'name': name,
            'repo_name': repo_name,
            'tag': tag,
            'stars': stars,
            'pulls': pulls,
            'distro': distro,
            'size': size,
            'os': os,
            'architecture': architecture,
            'sha_id': sha_id,
            'complete_size': complete_size,
            'layers': layers,
            'softwares': softwares,
            'nbr_layers': len(layers),
            'nbr_softwares': len(softwares),
        }

        repo_name_list[repo_name].append(repo_name_record)

    return repo_name_list


def load_images(path):
    docker_files = [filename for filename in listdir(path) if isfile(join(path, filename))]

    count = 0
    count_none = 0
    repo_name_list = defaultdict(list)

    for filename in sorted(docker_files, reverse=True)[1:]:
        if not filename.endswith('.json'):
            continue
        data = json.load(open(path + filename, 'r', encoding='utf8'))
        images = data['images']

        for image in images:
            df_id = image['_id']
            last_updated = image['last_updated']
            last_scan = image['last_scan']
            name = image['name']
            repo_name = image.get('repo_name', '')

            tag = image.get('tag', '')
            stars = image['stars']
            pulls = image['pulls']
            distro = image['distro']
            size = image['size']

            if distro is None or image['inspect_info'] is None or image['softwares'] is None:
                count_none += 1
                continue

            os = image['inspect_info']['Os']
            architecture = image['inspect_info']['Architecture']
            sha_id = image['inspect_info']['Id']
            complete_size = image['inspect_info']['Size']
            virtual_size = image['inspect_info']['VirtualSize']
            layers = image['inspect_info']['RootFS']['Layers']

            softwares = [(s['software'], s['ver']) for s in image['softwares']]

            repo_name_record = {
                'df_id': df_id,
                'last_updated': last_updated,
                'last_scan': last_scan,
                'name': name,
                'repo_name': repo_name,
                'tag': tag,
                'stars': stars,
                'pulls': pulls,
                'distro': distro,
                'size': size,
                'os': os,
                'architecture': architecture,
                'sha_id': sha_id,
                'complete_size': complete_size,
                'layers': layers,
                'softwares': softwares,
                'nbr_layers': len(layers),
                'nbr_softwares': len(softwares),
            }

            count += 1
            repo_name_list[repo_name].append(repo_name_record)

    return repo_name_list
    #print('Nbr repo', len(repo_name_list))
    #print('Nbr images', count, count_none)


def get_latest_as_repo(repo_name_list):
    repo_list = dict()
    last_updated = dict()

    for repo_name in repo_name_list:
        for image in repo_name_list[repo_name]:
            if image['tag'] != 'latest':
                continue
            repo = {
                'stars': image['stars'],
                'pulls': image['pulls'] / 1000000,
                'size': image['size'],
                'complete_size': image['complete_size'],
                'nbr_layers': image['nbr_layers'],
                'nbr_softwares': image['nbr_softwares'],
                'distro': image['distro'],
                'softwares': image['softwares'],
            }
            if image['repo_name'] not in repo_list:
                repo_list[image['repo_name']] = repo
                last_updated[image['repo_name']] = image['last_updated']
            elif last_updated[image['repo_name']] is not None and image['last_updated'] is not None and \
                    last_updated[image['repo_name']] < image['last_updated']:
                repo_list[image['repo_name']] = repo
                last_updated[image['repo_name']] = image['last_updated']
            # break
    return repo_list, last_updated


def dict2vector(reposet, target='pulls', use_software_version=True):
    y = list()
    X = list()
    byte2gigabyte = 1073741824
    distro_labels = dict()
    distro_labels_inverse = dict()
    software_labels = dict()
    software_labels_inverse = dict()
    names_list = list()

    for repo_name, repo in reposet.items():
        if repo['distro'] not in distro_labels:
            distro_labels[repo['distro']] = len(distro_labels)
            distro_labels_inverse[distro_labels[repo['distro']]] = repo['distro']

        for s in repo['softwares']:
            slabel = '%s%s' % (s[0], s[1]) if use_software_version else '%s' % (s[0])
            if slabel not in software_labels:
                software_labels[slabel] = len(software_labels)
                software_labels_inverse[software_labels[slabel]] = slabel

    for repo_name, repo in reposet.items():
        names_list.append(repo_name)
        distros = [0] * len(distro_labels)
        distros[distro_labels[repo['distro']]] = 1

        softwares = [0] * len(software_labels)
        for s in repo['softwares']:
            slabel = '%s%s' % (s[0], s[1]) if use_software_version else '%s' % (s[0])
            softwares[software_labels[slabel]] = 1

        # repo_vec = [repo['size'], repo['complete_size'], repo['nbr_layers'], repo['nbr_softwares'],
        #            distro_labels[repo['distro']]] + softwares

        repo_vec = [repo['size'] / byte2gigabyte,
                    repo['complete_size'] / byte2gigabyte,
                    repo['nbr_layers'],
                    repo['nbr_softwares']
                    ] + distros + softwares

        X.append(repo_vec)
        y.append(repo[target])

    features = ['size', 'csize', 'layers', 'softwares']
    features += sorted(distro_labels, key=distro_labels.get)
    features += sorted(software_labels, key=software_labels.get)

    les = (distro_labels, distro_labels_inverse, software_labels, software_labels_inverse, names_list, features)

    return np.array(X), np.array(y), les


def dict2vector_notarget(reposet, distro_labels, software_labels, use_software_version=True):
    X = list()

    for repo_name, repo in reposet.items():
        distros = [0] * len(distro_labels)
        distros[distro_labels[repo['distro']]] = 1

        softwares = [0] * len(software_labels)
        for s in repo['softwares']:
            slabel = '%s%s' % (s[0], s[1]) if use_software_version else '%s' % s
            softwares[software_labels[slabel]] = 1

        repo_vec = [repo['size'],
                    repo['complete_size'],
                    repo['nbr_layers'],
                    repo['nbr_softwares']
                    ] + distros + softwares

        X.append(repo_vec)

    return np.array(X)


def main():
    path = '../datasets/'
    repo_name_list = load_images(path)
    repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)

    repo_list = repo_list_latest
    X, y, les = dict2vector(repo_list, target='stars', use_software_version=False)


if __name__ == "__main__":
    main()

