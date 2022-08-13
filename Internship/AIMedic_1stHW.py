import pandas as pd

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
        if type(label) == list:
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