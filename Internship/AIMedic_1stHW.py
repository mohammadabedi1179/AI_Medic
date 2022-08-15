from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from sklearn.model_selection import KFold

class Preprocessing():

  def __init__(self, training_path, test_path):
    self.training_path = training_path
    self.test_path = test_path
    Preprocessing.__csv_to_dataframe__(self)

  def __csv_to_dataframe__(self):

    training_data = pd.read_csv(self.training_path)
    test_data = pd.read_csv(self.test_path)
    
    self.training_data = training_data
    self.test_data = test_data
  
  def count_features(self):

    training_data = self.training_data
    test_data = self.test_data
    datasets = [training_data, test_data]
    features = [{}, {}]
    c = 0

    for dataset in datasets:
      
      product_descriptions = []
      prices = []
      number_of_products = len(dataset['id'])

      for i in range(number_of_products):
        
        product = dataset['product_description'][i]
        product = product.replace("\\r","")
        product = product.replace("\\n", "")
        product = product.replace("\\u200c", "")
        product = product.replace("\\\\/", "")
        product = eval(product)
        
        for feature in product.keys():
          if feature not in features[c]:
            features[c][feature] = 0
          else:
            features[c][feature] += 1
        
      c += 1
      
    return features

  def extract_features(self, features):

    training_data = self.training_data
    test_data = self.test_data
    datasets = [training_data, test_data]
    data_dicts = []
    
    for dataset in datasets:
      
      product_descriptions = []
      prices = []
      number_of_products = len(dataset['id'])
      number_of_features = len(features)
      
      for i in range(number_of_products):
        
        product = dataset['product_description'][i]
        product = product.replace("\\r","")
        product = product.replace("\\n", "")
        product = product.replace("\\u200c", "")
        product = product.replace("\\\\/", "")
        product = eval(product)
    
        try:
          price = dataset['price'][i]
          
          if price != 0:
            prices.append(price)
            for feature in features:
              try:
                product_descriptions.append(product[feature])
              except:
                product_descriptions.append('نامعلوم')    
          
        except:
          for feature in features:
            try:
              product_descriptions.append(product[feature])
            except:
              product_descriptions.append('نامعلوم')
        
      
      if len(prices) != 0:
        data_dict = { 'price' : prices}
      else:
        data_dict = {}
      
      for i in range(number_of_features):
        data_dict[features[i]] = product_descriptions[i:number_of_features*number_of_products:number_of_features]   

      data_dicts.append(data_dict)

    return data_dicts

  def label_encoder(self, labels, train=True):
    if train:

      labels_dict = {}
      key = 0
      new_labels = []

      for label in labels:
        if type(label) != str:
          label = label[0]
        if label not in labels_dict.keys():
          labels_dict[label] = key
          key = key + 1
        new_labels.append(labels_dict[label])
      
      labels_dict['UNKNOWN'] = len(labels_dict.keys())
      self.labels_encoded_dict = labels_dict
    
    else:
      labels_dict = self.labels_encoded_dict
      new_labels = []
      for label in labels:
        try:
          new_labels.append(labels_dict[label])
        except:
          new_labels.append(labels_dict['UNKNOWN'])

    return new_labels     

class Regressor(Preprocessing):

  def __init__(self, training_path, test_path, features):
    
    Preprocessing.__init__(self, training_path, test_path)
    self.features = features
    self.data_dicts = Preprocessing.extract_features(self, self.features)
  
  def extract_arrays(self):
    data_dicts = self.data_dicts
    train_features = []
    test_features = []
    
    for feature in self.features:
      train_features.append(Preprocessing.label_encoder(self, data_dicts[0][feature]))
      test_features.append(Preprocessing.label_encoder(self, data_dicts[1][feature], train=False))
    train_features = np.array(train_features, dtype=np.int32).transpose()
    test_features = np.array(test_features, dtype=np.int32).transpose()
    train_labels = np.array(data_dicts[0]['price'], dtype=int).reshape(-1)
  
    return train_features, train_labels, test_features

  def plot_results(self, features_array, labels_array, font_dict):
    
    font = font_dict
    features = self.features
    number_of_features = len(features)
    plt.figure(figsize=(80, 80))
    
    for i in range(number_of_features):
        plt.subplot(int(np.ceil(np.sqrt(number_of_features))), int(np.ceil(np.sqrt(number_of_features))), i + 1)
        plt.scatter(features_array[:, i], labels_array)
        persian_labels = [f'قیمت اجناس بر حسب {features[i]}', 'قیمت به ریال', features[i]]
        persian_labels = [ get_display(reshape(persian_label)) for persian_label in persian_labels]
        plt.title(persian_labels[0], fontdict=font)
        plt.ylabel(persian_labels[1], fontdict=font)
        #plt.xlabel(persian_labels[2], fontdict=font)
        plt.xticks([0, max(features_array[:, i])])
    
    plt.show()

  def training(self, training_features, training_labels, test_features, model=LinearRegression(), num_of_cv_split=5, shuffle=True, output=True):
    kf = KFold(n_splits=num_of_cv_split, shuffle=shuffle)
    losses = []
    predictions = []

    for train_indices, valid_indices in kf.split(training_features):
        train_features, valid_features = training_features[train_indices], training_features[valid_indices]
        train_labels, valid_labels = training_labels[train_indices], training_labels[valid_indices]
        clf = model
        clf.fit(train_features, train_labels)
        predictions.append(clf.predict(test_features).reshape(-1))
        valid_predictions = clf.predict(valid_features).reshape(-1)
        loss = mean_absolute_percentage_error(valid_labels, valid_predictions)
        losses.append(loss)
        prediction = np.mean(predictions, axis=0)

        if output:
          id = np.arange(test_features.shape[0]).reshape(-1)
          output = {'id' : id, 'price' : prediction}
          output = pd.DataFrame.from_dict(output) 
          output.to_csv(r'Test.csv', index=False, header=True)
        
        return prediction, losses