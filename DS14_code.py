import pandas as pd
import numpy as np
import random
import math
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from scipy.stats import levene, ttest_ind
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr
from scipy.stats import skew, normaltest
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Setting the random seed
random_seed = 14276662 #Joz Zhou's ID

# prepare datasets
df_spotify = pd.read_csv('data/spotify52kData.csv')
df_ratings = pd.read_csv('data/starRatings.csv',header=None)
df_ratings.columns = df_spotify['track_name'].iloc[:5000] # manually specifying the index to be 5000 song names

# Checking for missing values and the data types of each column
missing_values = df_spotify.isnull().sum()
data_types = df_spotify.dtypes
# Overview of the dataset statistics
dataset_statistics = df_spotify.describe(include='all')
#it turns out the missing values are already handled in the spotify dataset
# data set overall is robust
#set sns for later uses
sns.set()
#global popularity data for later use
popularity_vals = np.asarray(df_spotify['popularity'])
#set basic skeleton for our question classes.
#we build classes for each question and build a run() funtion for them to facilitate testing each result/code
class QuestionInterface(object):
    @staticmethod
    def run():
        pass
# Q1: Is there a relationship between song length and popularity of a song? If so, is it positive or negative?
class Question1(QuestionInterface):
    @staticmethod
    def run():
        print('the number of missing values from the two associated columns are',
              df_spotify['popularity'].isnull().sum(),'and',df_spotify['duration'].isnull().sum())
        duration_vals = np.asarray(df_spotify['duration'])
        # plot histogram for duration
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.hist(duration_vals)
        plt.xlabel("song length in ms")
        plt.ylabel("frequency")
        plt.title('histogram for duration of the songs')
        # Plotting histogram of popularity
        plt.subplot(1, 2, 2)
        plt.hist(popularity_vals)
        plt.xlabel("popularity of a song")
        plt.ylabel("frequency")
        plt.title('histogram for popularity of a song')
        # test skewness and found the duration data is highly, positively skewed
        print(normaltest(duration_vals), normaltest(popularity_vals))
        # transform the data via natural log transformation to normalize it
        # so that we meet assumption for PearsonR test
        duration_logged = np.log(duration_vals)
        skew(duration_logged)
        # histogram for duration of songs after natural log transformation
        f, ax = plt.subplots()
        ax.hist(duration_logged)
        ax.set_xlabel("log(duration of songs)")
        ax.set_ylabel("frequency")
        ax.set_title("duration of songs after log transformation")
        qt = QuantileTransformer(n_quantiles=10, random_state=random_seed)
        duration_exp = duration_vals.reshape(-1, 1)
        transformed_duration = qt.fit_transform(duration_exp)
        print(normaltest(transformed_duration))
        x = transformed_duration.reshape(-1, )
        popularity_exp = popularity_vals.reshape(-1, 1)
        transformed_popularity = qt.fit_transform(popularity_exp)
        print(normaltest(transformed_popularity))
        y = transformed_popularity.reshape(-1, )
        reg1 = LinearRegression()
        reg1.fit(transformed_duration, y)
        y_hat = float(reg1.coef_[0]) * x + + reg1.intercept_
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', ms=2)
        plt.plot(x, y_hat, color='orange', linewidth=0.5)  # orange line for the fox
        plt.xlabel("song length")
        plt.ylabel("popularity of a song")
        plt.title("normalized song length vs. popularity of a song")
        plt.show()
        r2 = r2_score(y, y_hat)
        print('R^2:', r2.round(3))
        rmse = np.sqrt(np.mean(np.sum((y - y_hat) ** 2)))
        print('RMSE:', rmse.round(3))
        print(pearsonr(x, y))

# Question 2: Are explicitly rated songs more popular than songs that are not explicit?
class Question2(QuestionInterface):
    @staticmethod
    def run():
        # Check for missing values and data types in relevant columns
        missing_values = df_spotify[['explicit', 'popularity']].isnull().sum()
        data_types = df_spotify[['explicit', 'popularity']].dtypes
        print("missing values:",missing_values)
        print(data_types)
        # Descriptive statistics for both groups
        desc_stats_explicit = df_spotify[df_spotify['explicit'] == True]['popularity'].describe()
        desc_stats_non_explicit = df_spotify[df_spotify['explicit'] == False]['popularity'].describe()
        print(desc_stats_explicit)
        print(desc_stats_non_explicit)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='explicit', y='popularity', data=df_spotify)
        plt.title('Distribution of Popularity for Explicit and Non-Explicit Songs')
        plt.xlabel('Explicit')
        plt.ylabel('Popularity')
        plt.show()

        popularity_explicit = df_spotify[df_spotify['explicit'] == True]['popularity']
        popularity_non_explicit = df_spotify[df_spotify['explicit'] == False]['popularity']

        # Levene's Test for Homogeneity of variances
        levene_test_q2 = levene(popularity_explicit, popularity_non_explicit)
        print(f"Q2 levene-test result: {levene_test_q2}")

        # Perform Welch's t-test
        t_test_result = ttest_ind(popularity_explicit, popularity_non_explicit, equal_var=False)
        print(f"\nQ2 t-test result: {t_test_result}")
# Question 3: Are songs in major key more popular than songs in minor key?
class Question3(QuestionInterface):
    @staticmethod
    def run():
        # Visualize the distributions
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='mode', y='popularity', data=df_spotify)
        plt.title('Distribution of Popularity for Major and Minor Key Songs')
        plt.xlabel('Mode (0: Minor, 1: Major)')
        plt.ylabel('Popularity')
        plt.show()

        # Levene's test
        popularity_major = df_spotify[df_spotify['mode'] == 1]['popularity']
        popularity_minor = df_spotify[df_spotify['mode'] == 0]['popularity']
        levene_test_q3 = levene(popularity_major, popularity_minor)
        print(f"Q3 levene-test result: {levene_test_q3}")

        # Perform Welch's t-test again after re-importing necessary library
        t_test_result = ttest_ind(popularity_major, popularity_minor, equal_var=False, alternative = 'greater')
        print(f"\nQ3 t-test result: {t_test_result}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        # Q-Q plot for songs in major key
        stats.probplot(popularity_major, dist="norm", plot=ax1)
        ax1.set_title("Q-Q Plot for Popularity of Songs in Major Key", fontsize = 18)
        # Q-Q plot for songs in minor key
        stats.probplot(popularity_minor, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot for Popularity of Songs in Minor Key", fontsize = 18)
        plt.show()
# Q4: Which of the following 10 song features:
# duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness,
# liveness, valence and tempo predicts popularity best? How good is this model?
column_list = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness',
                   'valence', 'tempo']
class Question4(QuestionInterface):
    @staticmethod
    def run():
        # we do not transform data to reduce skewness for this question
        # because this kind of transformation may reduce R^2 for certain columns
        column = np.asarray(df_spotify['duration'])
        skewness1 = stats.skew(column)
        logged_col = np.log(column)
        skewness2 = stats.skew(logged_col)
        col_exp = column.reshape(-1, 1)
        logged_exp = logged_col.reshape(-1, 1)
        reg = LinearRegression()
        x = col_exp
        x_logged = logged_exp
        y = popularity_vals
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        scores = cross_val_score(reg, x, y, cv=kfold, scoring='r2').mean()
        scores_for_logged = cross_val_score(reg, x_logged, y, cv=kfold, scoring='r2').mean()
        print("skewness reduced from", skewness1, "to", skewness2, "while mean of R^2 dropped from", scores, "to",
              scores_for_logged)
        cols_dict = {}
        for i in column_list:
            cols_dict[i]=Question4.simple_linear(i)
        relevance = sorted(cols_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_list = [[i[0], i[1]] for i in relevance]
        print(sorted_list)
        table_q4 = pd.DataFrame(sorted_list, columns=['song feature', 'average COD via 10-fold'])
        print(table_q4)
        return(sorted_list)
    @staticmethod
    def simple_linear(column_name: str):
        column = np.asarray(df_spotify[column_name])
        col_exp = column.reshape(-1, 1)
        reg = LinearRegression()
        x = col_exp
        y = popularity_vals
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        scores = cross_val_score(reg, x, y, cv=kfold, scoring='r2')
        return scores.mean()
#function for later use
def get_sorted_list():
    cols_dict = {}
    for i in column_list:
        cols_dict[i]=Question4.simple_linear(i)
    relevance = sorted(cols_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_list = [[i[0], i[1]] for i in relevance]
    return sorted_list
# Q5: Building a model that uses *all* of the song features mentioned in question 4, how well can you predict popularity?
# How much (if at all) is this model improved compared to the model in question 4).
# How do you account for this? What happens if you regularize your model?
def make_x():
    temp_list = []
    for i in column_list:
        temp_list.append(np.asarray(df_spotify[i]).reshape(-1, 1))
    tuple_ten = tuple(temp_list)
    x = np.concatenate(tuple_ten, axis=1)
    return x
class Question5(QuestionInterface):
    @staticmethod
    def run():
        y = popularity_vals
        reg = LinearRegression()
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        x = make_x()
        scores = cross_val_score(reg, x, y, cv=kfold, scoring='r2')
        multi_score = scores.mean()
        print(multi_score)
        sorted_list = get_sorted_list()
        diff = multi_score - sorted_list[0][1]
        diff_percent = diff * 100 / sorted_list[0][1]
        print("compared to the best model in question 4, the R^2 increased by", str(diff.round(3)) + ',', 'achieving a',
              str(diff_percent.round(2)) + '%', "improvement")
        alphas = np.arange(0, 10, 0.5)
        best = [0, multi_score]
        alpha_dict = {}
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
            scores = cross_val_score(ridge, x, y, cv=kfold, scoring='r2')
            now_score = scores.mean()
            alpha_dict[alpha] = now_score
            if scores.mean() > best[1]:
                best = [alpha, scores.mean()]
        print(best)
        relevance = sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_list = [[i[0], i[1] - multi_score] for i in relevance]
        table_q5_1 = pd.DataFrame(sorted_list, columns=['alpha', 'improvement of R^2'])
        print(table_q5_1)
        print("improvement of",((best[1] - multi_score) / multi_score) * 100,"%")
        print(best[1] - multi_score)
        alphas = np.arange(0, 10, 0.5)
        best = [0, multi_score]
        alpha_dict = {}
        for alpha in alphas:
            lasso = Lasso(alpha=alpha)
            kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
            scores = cross_val_score(lasso, x, y, cv=kfold, scoring='r2')
            now_score = scores.mean()
            alpha_dict[alpha] = now_score
            if scores.mean() > best[1]:
                best = [alpha, scores.mean()]
        print(best)
        relevance = sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_list = [[i[0], i[1] - multi_score] for i in relevance]
        table_q5_2 = pd.DataFrame(sorted_list, columns=['alpha', 'improvement of R^2'])
        print(table_q5_2)
# Q6:When considering the 10 song features in the previous question, how many meaningful principal components can you extract?
# What proportion of the variance do these principal components account for?
# Using these principal components, how many clusters can you identify?
# Do these clusters reasonably correspond to the genre labels in column 20 of the data?
def get_PCA_transfored():
    x=make_x()
    zscoredData = stats.zscore(x)
    transformed = PCA(n_components=5).fit_transform(zscoredData)
    return transformed
class Question6(QuestionInterface):
    @staticmethod
    def run():
        # Z-score the data:
        x = make_x()
        zscoredData = stats.zscore(x)

        # Initialize PCA object and fit to our data:
        pca = PCA().fit(zscoredData)

        # Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
        eigVals = pca.explained_variance_

        # Rotated Data - simply the transformed data:
        # origDataNewCoordinates = pca.fit_transform(zscoredData)*-1
        numPredictors = np.size(x, axis=1)
        plt.bar(np.linspace(1, numPredictors, numPredictors), eigVals)
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.show()
        print('Proportion variance explained by the first 5 PCs:', np.sum(eigVals[:5] / np.sum(eigVals)).round(3))
        PCA_dict = {}
        for i in range(1, 11):
            PCA_dict[i] = np.sum(eigVals[:i] / np.sum(eigVals))
        PCA_list = [[i, PCA_dict[i]] for i in PCA_dict]
        table_q6 = pd.DataFrame(PCA_list, columns=['top n meaningful principal components',
                                                   'cumulative porportion of variance explained'])
        print(table_q6.style.hide())
        transformed = PCA(n_components=5).fit_transform(zscoredData)

        numClusters = 24  # loop over different # of clusters (2 to 25)
        Q = np.empty([numClusters, 1]) * np.NaN  # init container to store sums
        # Compute kMeans:
        for ii in tqdm(range(2, 2 + numClusters)):
            kMeans = KMeans(n_clusters=int(ii)).fit(transformed)  # compute kmeans using scikit
            cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
            cCoords = kMeans.cluster_centers_  # coordinate location for center of each cluster
            s = silhouette_samples(transformed, cId)  # compute the mean silhouette coefficient of all samples
            Q[ii - 2] = sum(s)  # take the sum
        plt.plot(np.arange(2, 26, 1), Q)
        plt.xticks(np.arange(2, 26, 1))
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of silhouette scores')
        plt.show()
        kmeans = KMeans(n_clusters=2, random_state=random_seed).fit(transformed)
        labels_kmeans = kmeans.labels_
        print(labels_kmeans)
        #conver genres to numeric values
        LE = LabelEncoder()
        genres = LE.fit_transform(df_spotify['track_genre'])
        print(np.unique(genres))#52 genres
        #below we make two lists of frequencies of genres from each cluster
        cluster_zero = np.asarray(labels_kmeans == 0).nonzero()[0]
        cluster_one = np.asarray(labels_kmeans == 1).nonzero()[0]
        fit_zero = np.zeros(52)
        fit_one = np.zeros(52)
        for i in cluster_zero:
            fit_index = genres[i]
            fit_zero[fit_index] += 1
        for i in cluster_one:
            fit_index = genres[i]
            fit_one[fit_index] += 1
        fit_zero = fit_zero * np.sum(fit_one) / np.sum(fit_zero)
        print(fit_zero)
        print(chisquare(fit_zero,fit_one))

# Question 7: Can you predict whether a song is in major or minor key from valence using logistic regression or a support vector machine?
# If so, how good is this prediction? If not, is there a better one?
class Question7(QuestionInterface):
    @staticmethod
    def run():
        # Setting the random seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        X = df_spotify[['valence']]
        y = df_spotify['mode']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        # Evaluating
        report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f'classification report: \n {report}')
        # Plotting the confusion matrix
        plt.figure(figsize=(4, 2))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Minor(0)', 'Major(1)'], yticklabels=['Minor(0)', 'Major(1)'],cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        # Check the balance of the classes (major vs. minor keys)
        class_balance = df_spotify['mode'].value_counts(normalize=True)
        print(f"class balance is: \n{class_balance}")

        major_key_songs = df_spotify[df_spotify['mode'] == 1]
        minor_key_songs = df_spotify[df_spotify['mode'] == 0]

        # Under-sampling the major key songs
        major_key_songs_sampled = major_key_songs.sample(n=len(minor_key_songs), random_state=random_seed)
        balanced_df = pd.concat([major_key_songs_sampled, minor_key_songs])
        balanced_df = shuffle(balanced_df, random_state=random_seed)
        X_balanced = balanced_df[['valence']]
        y_balanced = balanced_df['mode']
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=random_seed)

        log_reg_balanced = LogisticRegression()
        log_reg_balanced.fit(X_train_balanced, y_train_balanced)
        y_pred_balanced = log_reg_balanced.predict(X_test_balanced)

        # Evaluating
        accuracy_balanced = np.mean(y_test_balanced == y_pred_balanced)
        report_balanced = classification_report(y_test_balanced, y_pred_balanced)
        conf_matrix_balanced = confusion_matrix(y_test_balanced, y_pred_balanced)
        print(f'(balanced) classification report: \n {report_balanced}')
        # Plotting the confusion matrix
        plt.figure(figsize=(4, 2))
        sns.heatmap(conf_matrix_balanced, annot=True, fmt="d", cmap="Blues", xticklabels=['Minor(0)', 'Major(1)'], yticklabels=['Minor(0)', 'Major(1)'],cbar=False)
        plt.title('(Balanced) Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        # random forest on balanced dataset
        scaler_balanced = StandardScaler()
        X_train_balanced_scaled = scaler_balanced.fit_transform(X_train_balanced)
        X_test_balanced_scaled = scaler_balanced.transform(X_test_balanced)
        rf_balanced = RandomForestClassifier(random_state=random_seed)
        rf_balanced.fit(X_train_balanced_scaled, y_train_balanced)
        y_pred_rf_balanced = rf_balanced.predict(X_test_balanced_scaled)

        # Evaluating
        accuracy_rf_balanced = rf_balanced.score(X_test_balanced_scaled, y_test_balanced)
        conf_matrix_rf_balanced = confusion_matrix(y_test_balanced, y_pred_rf_balanced)
        class_report_rf_balanced = classification_report(y_test_balanced, y_pred_rf_balanced)
        print(f'(balanced) random forest report: \n {class_report_rf_balanced}')

        # Plotting the confusion matrix
        plt.figure(figsize=(4, 2))
        sns.heatmap(conf_matrix_rf_balanced, annot=True, fmt="d", cmap="Blues", xticklabels=['Minor(0)', 'Major(1)'], yticklabels=['Minor(0)', 'Major(1)'],cbar=False)
        plt.title('(Balanced) Random Forest Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

# Q8:Can you predict genre by using the 10 song features from question 4 directly
# or the principal components you extracted in question 6 with a neural network? How well does this work?
class Module(object):
    """
    Base class defining the structure and interface for neural network modules
    with placeholders for forward and backward computations.
    """

    def __init__(self):
        self.gradInput = None  # stores gradient
        self.output = None  # stores loss

    def forward(self, *input):
        """
        Placeholder for forward pass. Defines the computation performed at every call.
        Enforces that subclasses must implement their own version of the forward method
        """
        raise NotImplementedError

    def backward(self, *input):
        """
        Placeholder for backward pass. Defines the computation performed at every call.
        Enforces that subclasses must implement their own version of the backward method
        """
        raise NotImplementedError


class LeastSquareCriterion(Module):
    """
    This implementation of the least square loss assumes that the data comes as a 2 dimensional array
    of size (batch_size,num_classes) and the labels as a vector of size (num_classes)
    """

    def __init__(self, num_classes=52):
        super(LeastSquareCriterion, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, labels):
        target = np.zeros([x.shape[0], self.num_classes])
        for i in range(x.shape[0]):
            target[i, labels[i]] = 1
        self.output = np.sum((target - x) ** 2, axis=0)
        return np.sum(self.output)

    def backward(self, x, labels):
        self.gradInput = x
        for i in range(x.shape[0]):
            self.gradInput[i, labels[i]] = x[i, labels[i]] - 1  # gradient of loss
        return self.gradInput


class Linear(Module):
    """
    The input is supposed to have two dimensions (batchSize, in_feature)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features  # dimensions
        self.out_features = out_features  # dimensions
        np.random.seed(random_seed)
        self.weight = math.sqrt(1. / (out_features * in_features)) * np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)
        self.gradWeight = None
        self.gradBias = None

    def forward(self, x):  # this is our linear unit
        self.output = np.dot(x, self.weight.transpose()) + np.repeat(self.bias.reshape([1, -1]), x.shape[0], axis=0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = np.dot(gradOutput, self.weight)
        self.gradWeight = np.dot(gradOutput.transpose(), x)
        self.gradBias = np.sum(gradOutput, axis=0)
        return self.gradInput

    def gradientStep(self, lr):
        self.weight = self.weight - lr * self.gradWeight
        self.bias = self.bias - lr * self.gradBias


class ReLU(Module):
    """
    Implement the Rectified Linear Unit activation function for introducing non-linearity in the network.
    """

    def __init__(self, bias=True):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.output = x.clip(0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = (x > 0) * gradOutput
        return self.gradInput


class MLP(Module):
    """
    simple neural network architecture with two linear layers and a ReLU activation function in between.
    """

    def __init__(self, num_features=10, num_classes=52):
        super(MLP, self).__init__()
        self.fc1 = Linear(num_features, 64)
        self.relu1 = ReLU()
        self.fc2 = Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)  # notice: no relu on the final output
        return x

    def backward(self, x, gradient):
        gradient = self.fc2.backward(self.relu1.output, gradient)
        gradient = self.relu1.backward(self.fc1.output, gradient)
        gradient = self.fc1.backward(x, gradient)
        return gradient

    def gradientStep(self, lr):
        self.fc2.gradientStep(lr)
        self.fc1.gradientStep(lr)
        return True


def train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data, val_labels):
    n_train, n_val = len(train_data), len(val_data)
    train_loss = np.empty([num_epochs, int(n_train / batch_size)])
    val_loss = np.empty([num_epochs, int(n_val / batch_size)])

    for epoch in range(num_epochs):

        # Training loop
        for i in range(int(n_train / batch_size)):
            x = train_data[batch_size * i:batch_size * (i + 1)]
            y = train_labels[batch_size * i:batch_size * (i + 1)]
            y_pred = model.forward(x)
            train_loss[epoch, i] = criterion.forward(y_pred, y)
            grad0 = criterion.backward(y_pred, y)
            grad = model.backward(x, grad0)
            model.gradientStep(learn_rate)

            # Validation loop
        for j in range(int(n_val / batch_size)):
            x = val_data[batch_size * j:batch_size * (j + 1)]
            y = val_labels[batch_size * j:batch_size * (j + 1)]
            y_pred = model.forward(x)
            val_loss[epoch, j] = criterion.forward(y_pred, y)

        if (epoch + 1) % 10 == 0:
            print('Training epoch:', epoch + 1)

    # Plot output, if desired
    plt.plot(np.mean(train_loss, axis=1))
    plt.plot(np.mean(val_loss, axis=1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'])
    plt.show()


class CrossEntropyCriterion(Module):
    """
    This implementation of the cross-entropy loss assumes that the data comes as a 2 dimensional array
    of size (batch_size,num_classes) and the labels as a vector of size (num_classes)
    """

    def __init__(self, num_classes=10):
        super(CrossEntropyCriterion, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, labels):
        target = np.zeros([x.shape[0], self.num_classes])
        for i in range(x.shape[0]):
            target[i, labels[i]] = 1
        self.output = -np.sum(target * np.log(np.abs(x) + 1e-8))
        return self.output

    def backward(self, x, labels):
        self.gradInput = x
        for i in range(x.shape[0]):
            self.gradInput[i, labels[i]] = x[i, labels[i]] - 1
        return self.gradInput


def evaluate_model(model, val_data, val_labels, batch_size, num_samples):
    n_val = len(val_data)
    y_pred = np.empty([int(n_val / batch_size), batch_size])

    for i in range(int(n_val / batch_size)):
        x = val_data[batch_size * i:batch_size * (i + 1)]
        y = val_labels[batch_size * i:batch_size * (i + 1)]
        y_pred[i, :] = np.argmax(model.forward(x), axis=1)
    np.random.seed(random_seed)
    rand_index = np.random.randint(len(val_data), size=num_samples)
    model_accuracy = (y_pred.flatten()[rand_index] == val_labels[rand_index]).mean()

    return model_accuracy
class Question8(QuestionInterface):
    @staticmethod
    def run():
        temp_list = []
        for i in column_list:
            temp_list.append(np.asarray(df_spotify[i]).reshape(-1, 1))
        tuple_ten = tuple(temp_list)
        x = np.concatenate(tuple_ten, axis=1)
        x = stats.zscore(x)
        LE = LabelEncoder()
        genres = LE.fit_transform(df_spotify['track_genre'])
        #predict using 10 features
        num_epochs = 100
        learn_rate = 0.0005
        batch_size = 100
        num_classes = 52
        num_features = 10
        X_train, X_test, y_train, y_test = train_test_split(x, genres, test_size=0.2, random_state=random_seed)
        model = MLP(num_features, num_classes)
        criterion = CrossEntropyCriterion(num_classes)
        num_samples = int(1e3)
        train_data = X_train
        train_labels = y_train
        val_data = X_test
        val_labels = y_test
        train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data,
                    val_labels)
        model_accuracy = evaluate_model(model, val_data, val_labels, batch_size, num_samples)
        print('Model accuracy:', model_accuracy)
        num_epochs = 100
        learn_rate = 0.0005
        batch_size = 100
        num_classes = 52
        num_features = 5
        transformed = get_PCA_transfored()
        X_train, X_test, y_train, y_test = train_test_split(transformed, genres, test_size=0.2,
                                                            random_state=random_seed)
        model = MLP(num_features, num_classes)
        criterion = CrossEntropyCriterion(num_classes)
        num_samples = int(1e3)
        train_data = X_train
        train_labels = y_train
        val_data = X_test
        val_labels = y_test
        train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data,
                    val_labels)
        model_accuracy = evaluate_model(model, val_data, val_labels, batch_size, num_samples)
        print('Model accuracy:', model_accuracy)


# Question 9: In recommender systems, the popularity based model is an important baseline.
# We have a two part question in this regard: a) Is there a relationship between popularity and average star rating for the 5k songs we have explicit feedback for?
# b) Which 10 songs are in the “greatest hits” (out of the 5k songs), on the basis of the popularity based model?
# Q9a
class Question9(QuestionInterface):
    @staticmethod
    def run():
        average_ratings = df_ratings.mean(axis=0)
        df_spotify['average_rating'] = np.nan
        df_spotify.loc[:4999, 'average_rating'] = average_ratings.values

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='popularity', y='average_rating', data=df_spotify)
        plt.title('Relationship Between Popularity and Average Star Rating')
        plt.xlabel('Popularity')
        plt.ylabel('Average Star Rating')
        plt.show()

        # Spearman's rank correlation test
        spearman_corr, spearman_p_value = spearmanr(df_spotify['popularity'][:5000], df_spotify['average_rating'][:5000])
        spearman_corr, spearman_p_value
        print(f"Spearman's rank correlation between popularity and average star rating is {spearman_corr}, with a p-value of {spearman_p_value}")

        X_ols = df_spotify[['popularity']][:5000]
        y_ols = df_spotify['average_rating'][:5000]
        X_ols = sm.add_constant(X_ols)

        # OLS model
        ols_model = sm.OLS(y_ols, X_ols).fit()
        ols_summary = ols_model.summary()
        print(ols_summary)

        # Q9b
        sorted_songs = df_spotify[:5000].sort_values(by='average_rating', ascending=False)
        top_10_songs = sorted_songs.head(10)
        top_10_songs_details = top_10_songs[['artists', 'track_name', 'album_name', 'average_rating','popularity','track_genre']]
        print(top_10_songs_details)
        # here we explore the repeating entries, with conclusions in our written document
        # Finding the rows that are repeated based on both 'track_name' and 'album_name' and 'track_genre'
        duplicates = sorted_songs[sorted_songs.duplicated(subset=['track_name', 'album_name','track_genre'], keep=False)]
        sorted_duplicates = duplicates.sort_values(by=['track_name', 'album_name','track_genre'])
        sorted_duplicates[['songNumber','artists','album_name','track_name','popularity','average_rating']].head()

# Question 10: You want to create a “personal mixtape” for all 10k users who have explicit feedback for.
# This mixtape contains individualized recommendations as to the 10 songs (out of the 5k) a given user will enjoy most.
# How do these recommendations compare to the “greatest hits” from the previous question and how good is your recommender system in making recommendations?
### The SVD++ code might take 10 minutes to run ###
# Sample code to calculate Precision and Recall
def calculate_precision_recall(user_recommendations, df_ratings, threshold=3):
    precision_list = []
    recall_list = []
    for user_id, recommended_items in user_recommendations.items():
        # Actual liked items: items rated above the threshold
        actual_liked = set(df_ratings.columns[(df_ratings.iloc[user_id] >= threshold).fillna(False)])
        # Recommended items
        recommended = set(recommended_items)

        # Intersection of liked and recommended items
        relevant_and_recommended = recommended.intersection(actual_liked)

        # Precision and Recall calculations
        precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
        recall = len(relevant_and_recommended) / len(actual_liked) if actual_liked else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # Average Precision and Recall
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)

    return average_precision, average_recall

# predict ratings
def predict_ratings(user_similarity, ratings):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred

# Generate recommendations
class Question10(QuestionInterface):
    @staticmethod
    def run():
        reader = Reader(rating_scale=(0, 4))  # Assuming ratings are from 0 to 4
        data = Dataset.load_from_df(df_ratings.stack().reset_index(name='rating'), reader)
        trainset, testset = train_test_split(data, test_size=0.3, random_state=random_seed)
        ratings_matrix = df_ratings.fillna(-1).values
        # user-user cosine similarity
        user_similarity = cosine_similarity(ratings_matrix)
        predicted_ratings = predict_ratings(user_similarity, ratings_matrix)

        top_n_recommendations = 10
        user_recommendations = {}
        for user_id in range(predicted_ratings.shape[0]):
            user_unrated_items = np.where(df_ratings.iloc[user_id].isna())[0]
            user_predictions = predicted_ratings[user_id, user_unrated_items]
            top_items_indices = user_predictions.argsort()[-top_n_recommendations:][::-1]
            user_recommendations[user_id] = top_items_indices
        average_precision, average_recall = calculate_precision_recall(user_recommendations, df_ratings)
        print(f"average recall is: {average_recall}")
        print(f"average precision is: {average_precision}")

        all_recommended_songs = [song for recommendations in user_recommendations.values() for song in recommendations]
        song_frequency = Counter(all_recommended_songs)
        top_10_common_recommendations = [df_spotify.loc[song_id, 'track_name'] for song_id, _ in song_frequency.most_common(10)]
        print(top_10_common_recommendations)

        top_10_common_recommendations_indices = song_frequency.most_common(10)
        top_10_frequencies = [freq for _, freq in top_10_common_recommendations_indices]
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_common_recommendations, top_10_frequencies, color='skyblue')
        plt.xlabel('Songs')
        plt.ylabel('Counts of Recommendations')
        plt.title('Top 10 Most Frequently Recommended Songs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Extra Credit: Death metal and Children are two genres with great differences.
# The former is known for harshness, while the latter is known for harmony.
# We want to investigate whether the beats per measure differ between death metal and children music.
class ExtraCredit(QuestionInterface):
    @staticmethod
    def run():
        dealth_metal_df = df_spotify[df_spotify['track_genre'] == 'death-metal']
        children_df = df_spotify[df_spotify['track_genre'] == 'children']
        beats_death = np.asarray(dealth_metal_df['time_signature'])
        beats_children = np.asarray(children_df['time_signature'])
        f, ax = plt.subplots()
        ax.hist([beats_death, beats_children], label=['death metal', 'children'])
        ax.set_title("histogram for beats per measure from death metal and children songs")
        ax.set_xlabel("beats per measure")
        ax.set_ylabel("frequency")
        ax.legend()
        plt.show()
        print(mannwhitneyu(beats_death, beats_children))

if __name__ == '__main__':
    #below is a list with all problem classes. Running them thoroughly would print all results we obtained,
    # but the whole process would take ~20 minutes.
    #you can just run a part of them, or run any single question you like to test our results, or modify them on your own!
    problem_list = [Question1(),Question2(),Question3(),Question4(),Question5(),Question6(),Question7(),Question8(),Question9(),Question10(),ExtraCredit()]
    for each_question in problem_list:
        each_question.run()



