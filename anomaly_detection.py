import random
from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
from deep_river.anomaly import Autoencoder, ProbabilityWeightedAutoencoder, RollingAutoencoder
from river.compose import Pipeline
from river.preprocessing import MinMaxScaler
from torch import nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# HST
model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees(seed=42)
)
auc = metrics.ROCAUC()
random.seed(42)
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x, y)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("HST Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("HST Score: " + str(roc))

# SVM
model = anomaly.QuantileThresholder(
    anomaly.OneClassSVM(nu=0.2),
    q=0.995
)
auc = metrics.ROCAUC()
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("SVM Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("SVM Score: " + str(roc))


class MyAutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super(MyAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.non_linear = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_X):
        input_X = self.linear1(input_X)
        input_X = self.non_linear(input_X)
        input_X = self.linear2(input_X)
        return self.sigmoid(input_X)


ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
scaler = MinMaxScaler()
model = Pipeline(scaler, ae)
auc = metrics.ROCAUC(n_thresholds=50)
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x=x)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("AE Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("AE Score: " + str(roc))

wae = ProbabilityWeightedAutoencoder(module=MyAutoEncoder, lr=0.005)
model = Pipeline(scaler, wae)
auc = metrics.ROCAUC(n_thresholds=50)
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x=x)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("WAE Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("WAE Score: " + str(roc))


rae = RollingAutoencoder(module=MyAutoEncoder, lr=0.005)
model = Pipeline(scaler, rae)
auc = metrics.ROCAUC(n_thresholds=50)
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x=x)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("RAE Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("RAE Score: " + str(roc))


model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.ILOF()
)
auc = metrics.ROCAUC()
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x, y)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("ILOF Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("ILOF Score: " + str(roc))


model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.KitNet()
)
auc = metrics.ROCAUC()
scores, labels = [], []
for x, y in datasets.CreditCard().take(5000):
    score = model.score_one(x)
    model = model.learn_one(x, y)
    auc = auc.update(y, score)
    scores.append(score)
    labels.append(y)
print("KitNet Score: " + str(auc))
roc = roc_auc_score(labels, scores)
print("Kitnet Score: " + str(roc))


clf = anomaly.RobustRandomCutForest(num_trees=10, tree_size=25)
from tqdm import tqdm
scores, labels = [], []
auc = metrics.ROCAUC()
for x, y in tqdm(datasets.CreditCard().take(5000)):
    score = clf.score_one(x)
    clf.learn_one(x)
    scores.append(score)
    labels.append(y)
    auc = auc.update(y, score)

roc = roc_auc_score(labels, scores)
print("RRCF Score: " + str(auc))
print("RRCF Score: " + str(roc))
