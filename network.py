#Implement Simple CNN and train on character dataset
from parameters import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout


class CNN: 

    def __init__(self):
        # Define parameters
        self.h_input_shape      = (8,)                          #number of observations fed into the neural network
        self.h_intermediate_act = 'tanh'                        #activations function used in hidden layers
        self.h_output_act       = 'softmax'                     #activation used in the output layer
        self.h_optimizer        = 'adam'                        #optimizer used in compilation
        self.h_loss             = 'categorical_crossentropy'    #loss functions used in compilation
        self.h_metrics          = ['accuracy']                  #metric to track performance
        self.h_hidden_unit_count= 32                            #amount of hidden neurons in each hidden layer.
        self.h_dropout_rate     = 0.5                           #amount of dropout after each hidden layer
        self.h_output_neurons   = 3                             #number of output neurons, which should correspond to the number of actions possible in each state of the game
        self.model              = Sequential()
    

    #build the whole model and return it
    def build_model(self):
        model = Sequential()
        print("THIS IS THE MODEL SUMMARY:")
        print(model.summary())
        print("END MODEL SUMMARY.")

        #first layer
        model.add(Dense(self.h_hidden_unit_count, activation=self.h_intermediate_act, input_shape=self.h_input_shape))
        model.add(Dropout(self.h_dropout_rate))

        #second layer
        model.add(Dense(self.h_hidden_unit_count, activation=self.h_intermediate_act))
        model.add(Dropout(self.h_dropout_rate))

        #output layer
        model.add(Dense(self.h_output_neurons, activation=self.h_output_act))

        #compiling the whole model
        model.compile(loss=self.h_loss, optimizer=self.h_optimizer, metrics=self.h_metrics)
        return self.model