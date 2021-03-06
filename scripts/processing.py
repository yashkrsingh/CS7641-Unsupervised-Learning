import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, v_measure_score, silhouette_score, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn import preprocessing


def load_wine_data():
    wine = pd.read_csv('../data/winequality-white.csv', sep=',', header=0)
    bins = (0, 6, 10)
    labels = [0, 1]
    wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
    # print(wine['quality'].value_counts())
    return wine


def load_fetus_data():
    fetus = pd.read_csv('../data/fetal-health.csv', sep=',', header=0)
    col_to_drop = ['histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                   'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
                   'histogram_variance', 'histogram_tendency']
    fetus = fetus.drop(col_to_drop, axis=1)
    fetus["fetal_health"].replace({1: 0, 2: 1, 3: 1}, inplace=True)
    label_encoder = preprocessing.LabelEncoder()
    fetus["fetal_health"] = label_encoder.fit_transform(fetus["fetal_health"])
    # print(fetus['fetal_health'].value_counts())
    return fetus


def split_data_set(dataframe, seed):
    training_set, test_set = train_test_split(dataframe, train_size=0.8, shuffle=True, random_state=seed)
    train_x, train_y = training_set.iloc[:, :-1], training_set.iloc[:, -1]
    test_x, test_y = test_set.iloc[:, :-1], test_set.iloc[:, -1]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, train_y, test_x, test_y


def preprocess_data():
    datasets = {}
    wine = load_wine_data()
    fetus = load_fetus_data()

    dataframe = wine.sample(frac=1).reset_index(drop=True)
    x = dataframe.iloc[:, :-1]
    x = StandardScaler().fit_transform(x)
    y_label = dataframe.iloc[:, -1]
    datasets['wine'] = [x, y_label]

    dataframe = fetus.sample(frac=1).reset_index(drop=True)
    x = dataframe.iloc[:, :-1]
    x = StandardScaler().fit_transform(x)
    y_label = dataframe.iloc[:, -1]
    datasets['fetus'] = [x, y_label]

    return datasets


def perform_kmeans(dataset_name, method_name, x, y_label, seed, plot=False):
    k_max = 10
    kmeans_stats = pd.DataFrame(columns=['Name', 'k', 'Log Likelihood', 'Silhouette score', 'DB score', 'V measure'])
    results = pd.DataFrame(columns=['data', 'clustering', 'precision', 'recall', 'f1', 'accuracy'])

    for k in range(2, k_max):
        predictor = KMeans(n_clusters=k, random_state=seed)
        y = predictor.fit_predict(x)

        kmeans_stats.loc[kmeans_stats.shape[0]] = [dataset_name, k, predictor.score(x), silhouette_score(x, y),
                                                   davies_bouldin_score(x, y),
                                                   v_measure_score(y_label.to_numpy(dtype=int), y, beta=0)]
        kmeans_res = classification_report(y_label, y, output_dict=True)
        results.loc[results.shape[0]] = classification_scores(dataset_name, k, kmeans_res)

    dict_stats = {'Log Likelihood': kmeans_stats['Log Likelihood'],
                  'Silhouette score': kmeans_stats['Silhouette score'], 'DB score': kmeans_stats['DB score'],
                  'V measure': kmeans_stats['V measure']}
    plot_cluster_stats(dataset_name, 'kmeans', dict_stats)

    if method_name is not None:
        path = f'stats/exp3/{dataset_name}_{method_name}_kmeans_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kmeans_stats.to_csv(path, sep=',', encoding='utf-8')
    else:
        path = f'stats/exp1/{dataset_name}_kmeans_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kmeans_stats.to_csv(path, sep=',', encoding='utf-8')

        path = f'stats/exp1/{dataset_name}_kmeans_classification_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results.to_csv(path, sep=',', encoding='utf-8')

        plot_label_accuracy(f'{dataset_name} with KMeans', results)

    return dict_stats


def perform_em(dataset_name, method_name, x, y_label, seed, plot=False):
    n_max = 10
    em_stats = pd.DataFrame(columns=['Name', 'k', 'Log Likelihood', 'Silhouette score', 'DB score', 'V measure'])
    results = pd.DataFrame(columns=['data', 'clustering', 'precision', 'recall', 'f1', 'accuracy'])

    for n in range(2, n_max):
        predictor = GaussianMixture(n_components=n, random_state=seed)
        y = predictor.fit_predict(x)

        em_stats.loc[em_stats.shape[0]] = [dataset_name, n, predictor.score(x), silhouette_score(x, y),
                                           davies_bouldin_score(x, y),
                                           v_measure_score(y_label.to_numpy(dtype=int), y, beta=0)]
        em_res = classification_report(y_label, y, output_dict=True)
        results.loc[results.shape[0]] = classification_scores(dataset_name, n, em_res)

    dict_stats = {'Log Likelihood': em_stats['Log Likelihood'], 'Silhouette score': em_stats['Silhouette score'],
                  'DB score': em_stats['DB score'], 'V measure': em_stats['V measure']}
    plot_cluster_stats(dataset_name, 'em', dict_stats)

    if method_name is not None:
        path = f'stats/exp3/{dataset_name}_{method_name}_em_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        em_stats.to_csv(path, sep=',', encoding='utf-8')
    else:
        path = f'stats/exp1/{dataset_name}_em_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        em_stats.to_csv(path, sep=',', encoding='utf-8')

        path = f'stats/exp1/{dataset_name}_em_classification_results.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results.to_csv(path, sep=',', encoding='utf-8')

        plot_label_accuracy(f'{dataset_name} with EM', results)

    return dict_stats


def plot_label_accuracy(name, dataframe):
    dataframe.plot(x='clustering', y=['precision', 'recall', 'f1', 'accuracy'], kind='line', figsize=(10, 10))
    plt.title(f'Labelling Accuracy for {name}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Scores')
    plt.legend(loc='best')
    plt.grid()

    path = f'plots/exp1/{name}_labelling_accuracy.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_cluster_stats(dataset, method, stats):
    fig, axes = plt.subplots(1, len(stats), figsize=(20, 5))
    fig.suptitle(f'Num of clusters comparison for {dataset} and {method}')
    for i, score in enumerate(stats.items()):
        score_name, score_values = score
        ax = axes[i]
        ax.grid()
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel(score_name)
        x = list(range(2, len(score_values) + 2))
        ax.plot(x, score_values, label=score_name, lw=2)
        ax.set_xticks(x)
        ax.legend(loc='best')

    path = f'plots/exp1/{dataset}_{method}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_cluster_stats_batch(dataset, method, stats):
    n = len(stats['PCA'])
    fig, axes = plt.subplots(2, n - 2, figsize=(10, 10))
    faxes = axes.flatten()
    fig.suptitle(f'Cluster comparison for {dataset} with {method}')
    for k in stats:
        for i, score in enumerate(stats[k].items()):
            score_name, score_values = score
            ax = faxes[i]
            ax.grid()
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel(score_name)
            x = list(range(2, len(score_values) + 2))
            ax.plot(x, score_values, label=k, lw=2)
            ax.grid()
            ax.set_xticks(x)
            ax.legend(loc='best')

    path = f'plots/exp3/{dataset}_{method}_batch.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_transformations(dataset, method, title, x_label, y_label, values):
    fig, ax = plt.subplots()
    if title is not None:
        fig.suptitle(f'{title} for {dataset}')
    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    x = list(range(1, len(values) + 1))
    ax.plot(x, values)
    ax.set_xticks(x)

    path = f'plots/exp2/{dataset}_{method}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_reconstruction(dataset, dict_reconstruct, rp_reconstruct):
    fig, ax = plt.subplots()
    fig.suptitle(f'Reconstruction Error Comparison in {dataset} Dataset')
    ax.grid()
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Reconstruction Error')
    for method in dict_reconstruct:
        rec = dict_reconstruct[method]
        x = list(range(1, len(rec) + 1))
        ax.plot(x, rec, label=method)

    if rp_reconstruct is not None:
        mean, std = rp_reconstruct
        rec = mean
        x = list(range(1, len(rec) + 1))
        ax.plot(x, rec, label='RP')
        ax.fill_between(x, mean - std, mean + std, alpha=0.4, color='blue')

    ax.legend(loc='best')
    path = f'plots/exp2/reconstruction_{dataset}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_learning_curve(data_name, estimator, train_x, train_y, score_metric):
    plt.clf()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator, train_x, train_y, cv=5,
                                                                                    n_jobs=-1, return_times=True,
                                                                                    scoring=score_metric,
                                                                                    train_sizes=np.linspace(0.1, 1.0,
                                                                                                            5))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("F1 Score")
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title("Learning curve")

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-', label="Fit time")
    axes[1].plot(train_sizes, score_times_mean, 'o-', label="Score time")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha=0.1)

    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Time (sec)")
    axes[1].legend(loc="best")
    axes[1].set_title("Scalability of the model")

    path = f'plots/exp4-5/{data_name}_learningcurve.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')


def classification_scores(data, method, classification_report):
    precision = classification_report['macro avg']['precision']
    recall = classification_report['macro avg']['recall']
    f1 = classification_report['macro avg']['f1-score']
    accuracy = classification_report['accuracy']

    return [data, method, precision, recall, f1, accuracy]


def plot_color_map(df, data_name, groupby, x, y, title):
    n_clust = 2
    fig, ax = plt.subplots(figsize=(7, 7))

    cmap = plt.cm.get_cmap('viridis', 3)

    for i, cluster in df.groupby(groupby):
        cluster.plot(ax=ax, kind='scatter', x=x, y=y,
                     color=cmap(i / (n_clust - 1)),
                     label="%s %i" % (groupby, i),
                     alpha=0.3,
                     s=20)  # dot size
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title(title)

    path = f'plots/exp3/{data_name}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
