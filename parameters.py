#This file contains all program parameters


""" ###########
MAIN PARAMETERS
########### """
#M stands for main
M_NUM_GAMES             = 5                             #number of games to be played in total.
M_MAX_FRAMES_PER_GAME   = 1000                          #maximum amount of frames allowed per game, after which it will quit that game.



""" ##########
NETWORK PARAMS
########## """
#N stands for neural
N_INPUT_SHAPE           = (8,)                          #number of observations fed into the neural network
N_INTERMEDIATE_ACT      = 'tanh'                        #activations function used in hidden layers
N_OUTPUT_ACT            = 'softmax'                     #activation used in the output layer
N_OPTIMIZER             = 'adam'                        #optimizer used in compilation
N_LOSS                  = 'categorical_crossentropy'    #loss functions used in compilation
N_METRICS               = ['accuracy']                  #metric to track performance
N_HIDDEN_UNIT_COUNT     = 32                            #amount of hidden neurons in each hidden layer.
N_DROPOUT_RATE          = 0.5                           #amount of dropout after each hidden layer
N_OUTPUT_NEURONS        = 3                             #number of output neurons, which should correspond to the number of actions possible in each state of the game
