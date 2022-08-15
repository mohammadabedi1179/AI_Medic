from AIMedic_1stHW import Regressor, Preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

prep = Preprocessing('Internship/data/train.csv', 'Internship/data/test.csv')
features_count = prep.count_features()
sorted_tupels = []
for feature in features_count:
    sorted_tuple = list(sorted(feature.items(), key=lambda item: item[1]))
    sorted_tupels.append(sorted_tuple)

figure = plt.figure(figsize=(20, 20))
training_top_features = sorted_tupels[0][-10:]
test_top_features = sorted_tupels[1][-10:]
training_names = []
training_numbers = []

for feature in training_top_features:
  name, number = feature
  training_numbers.append(number)
  training_names.append(name)
persian_training_names = [get_display(reshape(persian_label)) for persian_label in training_names]
plt.subplot(2, 1, 1)
plt.bar(persian_training_names, training_numbers)

test_names = []
test_numbers = []

for feature in test_top_features:
  name, number = feature
  test_numbers.append(number)
  test_names.append(name)
persian_test_names = [get_display(reshape(persian_label)) for persian_label in test_names]
plt.subplot(2, 1, 2)
plt.bar(persian_test_names, test_numbers)
plt.show()
figure.savefig('out.png', format='png')

features = list(features_count[0].keys())
reg = Regressor('/content/drive/MyDrive/AIMedic/Internship/data/train.csv', '/content/drive/MyDrive/AIMedic/Internship/data/test.csv', features)
# extrcting features
training_features, training_labels, test_features = reg.extract_arrays()
#plotting results
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 54,
        }

#reg.plot_results(training_features, training_labels, font_dict=font)
predictions, losses = reg.training(training_features, training_labels, test_features, model=KNeighborsRegressor())
print(np.mean(losses))

models = [RandomForestRegressor(), KNeighborsRegressor(), DecisionTreeRegressor()]
mape_main_losses = []
mae_main_losses = []
mse_main_losses = []
for model in models:
  features = ['دسته بندی' ,'برند', 'وزن', 'ابعاد', 'مناسب برای', 'نوع اتصال', 'طول کابل', 'درگاههای ارتباطی', 'رابطها', 'قابلیتهای مقاومتی']
  reg = Regressor('/content/drive/MyDrive/AIMedic/Internship/data/train.csv', '/content/drive/MyDrive/AIMedic/Internship/data/test.csv', features)
  # extrcting features
  training_features, training_labels, test_features = reg.extract_arrays()
  predictions, mape_losses, mae_losses, mse_losses = reg.training(training_features, training_labels, test_features, model=RandomForestRegressor())
  mape_main_losses.append(np.mean(mape_losses))
  mae_main_losses.append(np.mean(mae_losses))
  mse_main_losses.append(np.mean(mse_losses))

fig = plt.figure(figsize=(10, 10))
plt.plot(['Ramdom Forest', 'KNN', 'Decision Tree'], np.log(mape_main_losses), label='MAPE')
plt.plot(['Ramdom Forest', 'KNN', 'Decision Tree'], np.log(mse_main_losses), label='MSE')
plt.plot(['Ramdom Forest', 'KNN', 'Decision Tree'], np.log(mae_main_losses), label='MAE')
plt.ylabel('Loss')
plt.xlabel('Model')
plt.legend(loc='upper right')
plt.show()
fig.savefig('out.png', format='png')