#Implement Simple CNN and train on character dataset
from parameters import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout


class DENSENET: 

    def __init__(self):
        # Define parameters
        self.h_input_shape      = N_INPUT_SHAPE                 #number of observations fed into the neural network
        self.h_intermediate_act = N_INTERMEDIATE_ACT            #activations function used in hidden layers
        self.h_output_act       = N_OUTPUT_ACT                  #activation used in the output layer
        self.h_optimizer        = N_OPTIMIZER                   #optimizer used in compilation
        self.h_loss             = N_LOSS                        #loss functions used in compilation
        self.h_metrics          = N_METRICS                     #metric to track performance
        self.h_hidden_unit_count= N_HIDDEN_UNIT_COUNT           #amount of hidden neurons in each hidden layer.
        self.h_dropout_rate     = N_DROPOUT_RATE                #amount of dropout after each hidden layer
        self.h_output_neurons   = N_OUTPUT_NEURONS              #number of output neurons, which should correspond to the number of actions possible in each state of the game
        self.model              = Sequential()
        print("densenet created... without building the model yet")


    #build the whole model and return it
    def build_model(self):
        model = Sequential()

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

        print("THIS IS THE MODEL SUMMARY:")
        print(model.summary())
        print("END MODEL SUMMARY.")

        return self.model