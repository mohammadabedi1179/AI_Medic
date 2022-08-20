from AIMedic_1stHW import Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
models = [RandomForestRegressor(), KNeighborsRegressor(), DecisionTreeRegressor()]
mape_main_losses = []
mae_main_losses = []
mse_main_losses = []
for model in models:
  features = ['دسته بندی' ,'برند', 'وزن', 'ابعاد', 'مناسب برای', 'نوع اتصال', 'طول کابل', 'درگاههای ارتباطی', 'رابطها', 'قابلیتهای مقاومتی']
  reg = Regressor('Internship/1st_HW/data/train.csv', 'Internship/1st_HW/data/test.csv', features)
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
plt.show()