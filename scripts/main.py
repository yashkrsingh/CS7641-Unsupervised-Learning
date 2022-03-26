from scipy.linalg import pinv
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.random_projection import GaussianRandomProjection

from processing import *


def part1():
    seed = 42
    np.random.seed(seed)

    datasets = preprocess_data()

    # K Means
    for dataset_name, data in datasets.items():
        x = data[0]
        y_label = data[1]

        perform_kmeans(dataset_name, None, x, y_label, seed, plot=True)

    # Expectation Maximization
    for dataset_name, data in datasets.items():
        x = data[0]
        y_label = data[1]

        perform_em(dataset_name, None, x, y_label, seed, plot=True)


def part2():
    seed = 42
    np.random.seed(seed)

    datasets = preprocess_data()

    for dataset_name, data in datasets.items():
        x = data[0]
        y_label = data[1]
        dict_reconstruct = {'PCA': [], 'ICA': [], 'MI': []}

        pca = PCA()
        new_x = pca.fit_transform(x)
        plot_transformations(dataset_name, 'pca_exp_variance', 'PCA Explained Variance', 'Features',
                             'Total Explained variance',
                             pca.explained_variance_ratio_.cumsum())
        avg_kurtosis = []

        for k in range(1, x.shape[1] + 1):
            pca = PCA(n_components=k)
            new_x = pca.fit_transform(x)
            projection = pca.inverse_transform(new_x)
            reconstruct_err = np.mean(np.square(x - projection))
            dict_reconstruct['PCA'].append(reconstruct_err)
            abs_val = np.abs(kurtosis(new_x)).mean()
            avg_kurtosis.append(abs_val)
        plot_transformations(dataset_name, 'pca_kurt', 'Kurtosis (PCA)', 'Dimensions', 'Mean Kurtosis', avg_kurtosis)

        avg_kurtosis = []
        for k in range(1, x.shape[1] + 1):
            ica = FastICA(n_components=k)
            new_x = ica.fit_transform(x)
            projection = ica.inverse_transform(new_x)
            reconstruct_err = np.mean(np.square(x - projection))
            dict_reconstruct['ICA'].append(reconstruct_err)
            val = np.abs(kurtosis(new_x)).mean()
            avg_kurtosis.append(val)
        plot_transformations(dataset_name, 'ica_kurt', 'Kurtosis (ICA)', 'Dimensions', 'Mean Kurtosis', avg_kurtosis)

        rand_proj_rec = []
        for i in range(1, 5):
            curr_rp_rec = []
            for k in range(1, x.shape[1] + 1):
                rp = GaussianRandomProjection(n_components=k)
                new_x = rp.fit_transform(x)
                projection = (pinv(rp.components_) @ new_x.T).T
                reconstruct_err = np.mean(np.square(x - projection))
                curr_rp_rec.append(reconstruct_err)
            rand_proj_rec.append(curr_rp_rec)
        rand_proj_rec = np.asarray(rand_proj_rec)
        rand_proj_rec = (rand_proj_rec.mean(axis=0), rand_proj_rec.std(axis=0))

        mi = SelectKBest(mutual_info_classif, k=x.shape[1])
        new_x = mi.fit_transform(x, y_label)
        plot_transformations(dataset_name, 'mi', 'Mututal Information', 'Features (asc)', 'Mutual Information',
                             sorted(mi.scores_))
        for k in range(1, x.shape[1] + 1):
            mi = SelectKBest(mutual_info_classif, k=k)
            new_x = mi.fit_transform(x, y_label)
            projection = mi.inverse_transform(new_x)
            reconstruct_err = np.mean(np.square(x - projection))
            dict_reconstruct['MI'].append(reconstruct_err)

        plot_reconstruction(dataset_name, dict_reconstruct, rand_proj_rec)


def part3():
    seed = 42
    k = 6
    np.random.seed(seed)

    datasets = preprocess_data()

    for dataset_name, data in datasets.items():
        x = data[0]
        y_label = data[1]

        stats = {'kmeans': {}, 'em': {}}
        for method_name, fs_method in {
            'PCA': PCA(k),
            'ICA': FastICA(k),
            'RP': GaussianRandomProjection(k),
            'MI': SelectKBest(mutual_info_classif, k=k)}.items():
            new_x = fs_method.fit_transform(x, y_label)
            k_stats = perform_kmeans(dataset_name, method_name, new_x, y_label, seed, plot=False)
            e_stats = perform_em(dataset_name, method_name, new_x, y_label, seed, plot=False)
            stats['kmeans'][method_name] = k_stats
            stats['em'][method_name] = e_stats

        plot_cluster_stats_batch(f'{dataset_name}', 'kmeans', stats['kmeans'])
        plot_cluster_stats_batch(f'{dataset_name}', 'em', stats['em'])


if __name__ == '__main__':
    part1()
    part2()
    part3()
