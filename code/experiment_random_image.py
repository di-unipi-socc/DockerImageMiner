import datetime
import warnings

from darter import *
from docker_manager import *
from docker_generator import *

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# filename_rnd_img = 'rnd_img_uniform_%s.csv'
filename_rnd_img = 'rnd_img_%s.csv'
header_rnd_img = 'iter,image_id,target,ispopular\n'


def main():
    path = '../datasets/'
    path_exp = '../experiments/'
    target = 'pulls'

    repo_name_list = load_images(path)
    repo_list_latest, last_updated = get_latest_as_repo(repo_name_list)

    repo_list = repo_list_latest
    size, csize, nlayers, nsoftwares, distro, softwares, softwarep = get_prob_distributions(repo_list, n=1000)
    # size, csize, nlayers, nsoftwares, distro, softwares = get_uniform_prob_distributions(repo_list, n=1000)

    repo_list = repo_list_latest
    X, y, les = dict2vector(repo_list, target=target, use_software_version=False)

    distro_labels = les[0]
    software_labels = les[2]

    file_rnd_img = open(path_exp + (filename_rnd_img % target), 'w')
    file_rnd_img.write(header_rnd_img)

    nbr_iter = 10
    nbr_rnd_img = 10000
    for iter in range(0, nbr_iter):
        print('Iteration %d' % iter)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=iter)
        darter_est = DARTER()
        darter_est.fit(X_train, y_train)
        random_images = generate_random_images(size, csize, nlayers, nsoftwares, distro, softwares, softwarep,
                                              nbr_images=nbr_rnd_img)
        # random_images = generate_uniform_random_images(size, csize, nlayers, nsoftwares, distro, softwares,
        #                                                nbr_images=nbr_rnd_img)
        X_test = dict2vector_notarget(random_images, distro_labels, software_labels, use_software_version=False)
        y_pred = darter_est.predict(X_test)
        for image_id, target in enumerate(y_pred):
            ispopular = 1 if target >= darter_est.success_threshold else 0
            file_rnd_img.write('%s,%s,%s,%d\n' % (iter, image_id, target, ispopular))
        file_rnd_img.flush()
    file_rnd_img.close()


if __name__ == "__main__":
    main()
