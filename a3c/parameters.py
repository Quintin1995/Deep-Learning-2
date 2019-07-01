# Parameter file for a3c

""" ###########
MAIN PARAMETERS
########### """
#M stands for main
GAME 					= "Breakout-v0"					# Self explanatory, name of the game
M_NUM_GAMES             = 1000                          #number of games to be played in total.
M_MAX_FRAMES_PER_GAME   = 300                           #maximum amount of frames allowed per game, after which it will quit that game.
M_DO_RENDER_GAME        = True                          #set to true if you want to see the visuals of the game
M_PLAY_BATCH_SIZE       = 8                             #an agent replays this many state  games from stored states. size of how many games are played
M_RENDER_GAME_MODULO    = 1                            	# Render every Nth game


""" ##########
RESULTS PARAMETERS
############## """
#R stands for results
R_AVG_RANGE     = 10                            # Take the average result over N individual game results
R_PLOTS_PATH    = 'plots/'                      # Directory to place the plots in
R_PLOTS_FILE    = 'reward_results.eps'          # File name for result plots

""" ##########
NETWORK PARAMS
########## """
N_POOL_DIM				= (2,2)							# Size of the max pooling pass for convolutional block
N_CONV_DIM				= 32							# Number of hidden units in convolutional layer
N_DENSE_DIM				= 100							# Number of hidden units in a dense layer

""" ##########
ACTOR CRITIC PARAMS
###########"""
A_LEARN_RATE			= 0.0001						# Learning rate
A_GAMMA					= 0.99							# Gamma for Q-parameter
A_MAX_EPS				= 200							# Number of epochs
A_UPDATE_FREQ			= 20							# Number of time steps between model updates


#N stands for neural
N_INPUT_DIM             = 8                             #number of observations fed into the neural network
N_INTERMEDIATE_ACT      = 'tanh'                        #activations function used in hidden layers
N_OUTPUT_ACT            = 'softmax'                     #activation used in the output layer
N_OPTIMIZER             = 'adam'                        #optimizer used in compilation

#options['mse', 'categorical_crossentropy']
N_LOSS                  = 'categorical_crossentropy'    #loss functions used in compilation
N_METRICS               = ['accuracy']                  #metric to track performance
N_MODEL_FILE_PATH       = 'saved_models/'               #path for saving the model
N_MODEL_FILE_NAME       = 'a3c.h5'                    	#file name for saving the model

