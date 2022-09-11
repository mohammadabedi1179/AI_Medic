from captcha_project import NN, History, Results
import tensorflow as tf
import pickle
import keras
if __name__=="__main__":
  net = NN()
  training, validation, test = net.load_data("AI_Medic/Internship/CAPTCHA_project/data", 32, 0.75, 0.15)
  labels_dict = net.labels_dict
  words_dict = dict((v, k) for k, v in labels_dict.items())
  model = keras.models.load_model('AI_Medic/Internship/CAPTCHA_project/saved model/new', compile=False)
  with open('/content/Final_history.pkl', 'rb') as f:
        new_history = pickle.load(f)
        history = History(new_history)
  command = ''
  while command != 'exit':
    command = input("Please insert your command:" + "\n" + "insert number 1 to plot training loss and accuracy history" +  "\n" + "insert number 2 to plot some model predictions on test dataset"+ "\n" + "and insert number 3 to start a new training process" + "\n")    
    if command == '1':
      re = Results(history, model)
      re.plot_loss_accuracy()
    elif command == '2':
      re = Results(history, model)
      re.test(test)
    elif command == '3':
      model = net.load_model()
      epochs = int(input('please enter number of epochs (at least 100 would be desired)'))
      history = model.fit(training,validation_data=validation, epochs=200, callbacks=[net.call_back])
    else:
      print('please insert right command')