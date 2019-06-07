#This file contains all program parameters


""" ###########
MAIN PARAMETERS
########### """
#M stands for main
M_NUM_GAMES             = 50                            #number of games to be played in total.
M_MAX_FRAMES_PER_GAME   = 200                           #maximum amount of frames allowed per game, after which it will quit that game.
M_DO_RENDER_GAME        = True                          #set to true if you want to see the visuals of the game
M_PLAY_BATCH_SIZE       = 8                             #an agent replays this many state  games from strored states. size of how many games are played
M_RENDER_GAME_MODULO    = 10                            # Render every Nth game

""" ##########
NETWORK PARAMS
########## """
#N stands for neural
N_INPUT_DIM             = 8                             #number of observations fed into the neural network
N_INTERMEDIATE_ACT      = 'tanh'                        #activations function used in hidden layers
N_OUTPUT_ACT            = 'softmax'                     #activation used in the output layer
N_OPTIMIZER             = 'adam'                        #optimizer used in compilation
#options['mse', 'categorical_crossentropy']
N_LOSS                  = 'categorical_crossentropy'    #loss functions used in compilation
N_METRICS               = ['accuracy']                  #metric to track performance
N_HIDDEN_UNIT_COUNT     = 32                            #amount of hidden neurons in each hidden layer.
N_DROPOUT_RATE          = 0.5                           #amount of dropout after each hidden layer
N_OUTPUT_NEURONS        = 4                             #number of output neurons, which should correspond to the number of actions possible in each state of the game
N_MODEL_FILE_PATH       = 'saved_models/'               #path for saving the model
N_MODEL_FILE_NAME       = 'model.h5'                    #file name for saving the model


""" #############
Q_LEARNING PARAMS
############# """
#all parameters that have to do with q-learning
Q_MAX_EPSILON           = 1.0                           #maximum value for a random action in the search space
Q_MIN_EPSILON           = 0.05                          #minimum probability for a random chance in search space (example if 0.05, then there is a 5% chance of taking an exploratory move)
Q_DECAY_EPSILON         = 0.995                         #The maximum epsilon value is multiplied by this number to get a lower epsilon value. (so less chance of a exploratory move happing as max epsilon decreases)
Q_MAX_STATES_RETAINED   = 1000                          #The maximum number of states stored in the agents state memory list
Q_GAMMA                 = 0.99                          #The discount value of state. How much the valuation of state is discounted. Gamma determines the importance of future rewards.
Q_LEARNING_RATE         = 0.001                         #Learning rate for the q-learning algorithm