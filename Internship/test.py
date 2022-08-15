from AIMedic_1stHW import Regssor, Preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

features = ['دسته بندی' ,'برند', 'وزن', 'ظرفیت', 'کارکرد', 'جنس بدنه', 'پردازنده', 'درگاه', 'تعداد پورت', 'سازگار با', 'دستگاه های سازگار', 'سازنده پردازنده', 'سیستم عامل', 'ابعاد', 'نوع پورت', 'منبع تغذیه', 'جنس', 'ولتاژ', 'مناسب برای', 'نوع حافظه', 'مقدار حافظه', 'نسبت تصویر', 'حداکثر وزن قابل تحمل', 'نوع کابل', 'مدل پردازنده', 'سری پردازنده']
reg = Regssor('data/train.csv', 'data/test.csv', features)
# extrcting features
training_features, training_labels, test_features = reg.extract_arrays()

#plotting results
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 8,
        }
#reg.plot_results(training_features, training_labels, font_dict=font)
predictions, losses = reg.training(training_features, training_labels, test_features, model=KNeighborsRegressor(n_neighbors=4))
print(np.mean(losses))

