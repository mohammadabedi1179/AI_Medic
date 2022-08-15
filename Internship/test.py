from AIMedic_1stHW import Regssor, Preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

"""features = ['دسته بندی' ,'برند', 'وزن', 'کارکرد', 'سری پردازنده']
prep = Preprocessing('data/train.csv', 'data/test.csv')
features = prep.count_features()
for feature in features:
    print(dict(sorted(feature.items(), key=lambda item: item[1])))"""

features = ['دسته بندی' ,'برند', 'وزن', 'ابعاد', 'مناسب برای', 'نوع اتصال', 'منبع تغذیه', 'درگاههای ارتباطی', 'ظرفیت']
reg = Regssor('data/train.csv', 'data/test.csv', features[:4    ])
# extrcting features
training_features, training_labels, test_features = reg.extract_arrays()
#plotting results
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 8,
        }

reg.plot_results(training_features, training_labels, font_dict=font)
predictions, losses = reg.training(training_features, training_labels, test_features, model=DecisionTreeRegressor())
print(np.mean(losses))