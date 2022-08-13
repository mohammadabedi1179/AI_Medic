from AIMedic_1stHW import Preprocessing
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from sklearn.model_selection import KFold


prep = Preprocessing('/home/mohammadabedi/Desktop/data/train.csv', '/home/mohammadabedi/Desktop/data/test.csv')
features = ['برند', 'دسته بندی', 'وزن', 'کارکرد', 'پورت', 'منبع تغذیه']
data_dicts = prep.extract_features(features)
train_features = []
test_features = []
for feature in features:
    train_features.append(prep.label_encoder(data_dicts[0][feature]))
    test_features.append(prep.label_encoder(data_dicts[1][feature], train=False))

train_features = np.array(train_features, dtype=np.int32).transpose()
test_features = np.array(test_features, dtype=np.int32).transpose()
train_labels = np.array(data_dicts[0]['price'], dtype=int).reshape(-1)

# plot the features
font = {'family': 'B Nazanin',
        'weight': 'normal',
        'size': 16,
        }
number_of_features = len(features)
plt.figure(figsize=(80, 10))
for i in range(number_of_features):
    plt.subplot(number_of_features, 1, i + 1)
    plt.scatter(train_features[:, i], train_labels)
    persian_labels = [f'قیمت اجناس بر حسب {features[i]}', 'قیمت به ریال', features[i]]
    persian_labels = [ get_display(reshape(persian_label)) for persian_label in persian_labels]
    plt.title(persian_labels[0], fontdict=font)
    plt.ylabel(persian_labels[1], fontdict=font)
    plt.xlabel(persian_labels[2], fontdict=font)
plt.show()
kf = KFold(shuffle=True)
for train_indices, test_indices in kf.split(train_features):
    training_features, valid_features = train_features[train_indices], train_features[test_indices]
    training_labels, valid_labels = train_labels[train_indices], train_labels[test_indices]
    clf = DecisionTreeRegressor(min_samples_split=5)
    #clf = RandomForestRegressor()
    clf.fit(training_features, training_labels)
    predictions = clf.predict(test_features).reshape(-1)
    id = np.arange(34262).reshape(-1)
    output = {'id' : id, 'price' : predictions}
    output = pd.DataFrame.from_dict(output)
    output.to_csv(r'output.csv', index=False, header=True)
    predictions = clf.predict(valid_features).reshape(-1)
    loss = mean_squared_log_error(valid_labels, predictions)
    print(loss) 