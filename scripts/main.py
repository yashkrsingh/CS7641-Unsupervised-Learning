import numpy as np
from scipy.linalg import pinv
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
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
            pca = PCA(n_components=k, random_state=seed)
            new_x = pca.fit_transform(x)
            projection = pca.inverse_transform(new_x)
            reconstruct_err = np.mean(np.square(x - projection))
            dict_reconstruct['PCA'].append(reconstruct_err)
            abs_val = np.abs(kurtosis(new_x)).mean()
            avg_kurtosis.append(abs_val)
        plot_transformations(dataset_name, 'pca_kurt', 'Kurtosis (PCA)', 'Dimensions', 'Mean Kurtosis', avg_kurtosis)

        avg_kurtosis = []
        for k in range(1, x.shape[1] + 1):
            ica = FastICA(n_components=k, random_state=seed)
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
                rp = GaussianRandomProjection(n_components=k, random_state=seed)
                new_x = rp.fit_transform(x)
                projection = (pinv(rp.components_) @ new_x.T).T
                reconstruct_err = np.mean(np.square(x - projection))
                curr_rp_rec.append(reconstruct_err)
            rand_proj_rec.append(curr_rp_rec)
        rand_proj_rec = np.asarray(rand_proj_rec)
        rand_proj_rec = (rand_proj_rec.mean(axis=0), rand_proj_rec.std(axis=0))

        mi = SelectKBest(mutual_info_classif, k=x.shape[1])
        new_x = mi.fit_transform(x, y_label)
        plot_transformations(dataset_name, 'mi', 'Mutual Information', 'Features (asc)', 'Mutual Information',
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
            'PCA': PCA(k, random_state=seed),
            'ICA': FastICA(k, random_state=seed),
            'RP': GaussianRandomProjection(k, random_state=seed),
            'MI': SelectKBest(mutual_info_classif, k=k)}.items():
            new_x = fs_method.fit_transform(x, y_label)
            k_stats = perform_kmeans(dataset_name, method_name, new_x, y_label, seed, plot=False)
            e_stats = perform_em(dataset_name, method_name, new_x, y_label, seed, plot=False)
            stats['kmeans'][method_name] = k_stats
            stats['em'][method_name] = e_stats

        plot_cluster_stats_batch(f'{dataset_name}', 'kmeans', stats['kmeans'])
        plot_cluster_stats_batch(f'{dataset_name}', 'em', stats['em'])


def part4():
    seed = 42
    k = 6
    np.random.seed(seed)

    wine = load_wine_data()
    wine_train_x, wine_train_y, wine_test_x, wine_test_y = split_data_set(wine, seed)

    results = pd.DataFrame(columns=['data', 'method', 'precision', 'recall', 'f1', 'accuracy'])

    for method_name, fs_method in {
        'PCA': PCA(k, random_state=seed),
        'ICA': FastICA(k, random_state=seed),
        'RP': GaussianRandomProjection(k, random_state=seed),
        'MI': SelectKBest(mutual_info_classif, k=k)}.items():
        new_train_x = fs_method.fit_transform(wine_train_x, wine_train_y)
        new_test_x = fs_method.fit_transform(wine_test_x, wine_test_y)

        nn = MLPClassifier(activation='relu', hidden_layer_sizes=[200], learning_rate_init=.001,
                           learning_rate='adaptive',
                           max_iter=10000)
        nn.fit(new_train_x, wine_train_y)
        prediction = nn.predict(new_test_x)
        print(confusion_matrix(wine_test_y, prediction))
        wine_train_result = classification_report(wine_test_y, prediction, output_dict=True)

        plot_learning_curve(f'wine_{method_name}', nn, new_train_x, wine_train_y, 'f1')
        results.loc[results.shape[0]] = classification_scores('wine', method_name, wine_train_result)

    results.to_csv('nn_fs.csv', sep=',', encoding='utf-8')


def part5():
    seed = 42
    k = 2
    np.random.seed(seed)

    wine = load_wine_data()
    wine_train_x, wine_train_y, wine_test_x, wine_test_y = split_data_set(wine, seed)

    results = pd.DataFrame(columns=['data', 'clustering', 'precision', 'recall', 'f1', 'accuracy'])

    for predictor_name, pr_method in {
        'kmeans': KMeans(n_clusters=k, random_state=seed),
        'em': GaussianMixture(n_components=k, random_state=seed)}.items():

        # pr_method.fit(wine_train_x)
        train_clusters = pr_method.fit_predict(wine_train_x, wine_train_y)
        train_x_clusters = wine_train_x.copy()
        train_x_clusters = np.column_stack((train_x_clusters, train_clusters))

        test_clusters = pr_method.predict(wine_test_x)
        test_x_clusters = wine_test_x.copy()
        test_x_clusters = np.column_stack((test_x_clusters, test_clusters))

        nn = MLPClassifier(activation='relu', hidden_layer_sizes=[200], learning_rate_init=.001,
                           learning_rate='adaptive',
                           max_iter=10000)
        nn.fit(train_x_clusters, wine_train_y)
        prediction = nn.predict(test_x_clusters)
        print(confusion_matrix(wine_test_y, prediction))
        wine_train_result = classification_report(wine_test_y, prediction, output_dict=True)

        plot_learning_curve(f'wine_{predictor_name}', nn, train_x_clusters, wine_train_y, 'f1')
        results.loc[results.shape[0]] = classification_scores('wine', predictor_name, wine_train_result)

    results.to_csv('nn_clustering.csv', sep=',', encoding='utf-8')


if __name__ == '__main__':
    # part1()
    # part2()
    # part3()
    # part4()
    part5()
