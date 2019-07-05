# Base imports
import argparse

import dqn.dqn as dqn
import a3c
# import other frameworks here

valid_alg = [
    'dqn',
    'a3c',
    'pdqn']

def main():
    parser = argparse.ArgumentParser()
    mutex1 = parser.add_mutually_exclusive_group()
    # This means we can now either train or validate. Validate is just here so we can have single letter options because train and test start with the same letter
    #mutex1.add_argument('-t',type=str,help="Flag for training one of the methods. Takes one positional argument (being the name of the method).")
    mutex1.add_argument('-v',type=str,help="Flag for validation/testing one of the methods. Takes one positional argument.")
    parser.add_argument('--norender',help="Disables rendering the game during training/testing so we can run it on peregrine.")
    parser.add_argument('--double', action='store_true', help="Enables frozen weights DOUBLE network for DQN model.")
    parser.add_argument('--duel', action='store_true',help="Uses a dueling network structure when running with a DQN network.")
    parser.add_argument('--epochs', type=int, help="Specify amount of epochs to run.")
    parser.add_argument('--memory', type=int, help="Specify amount of experiences we can store at once.")
    parser.add_argument('--replay_batch_size', type=int, help="Specify amount of experiences to replay per replay session.")
    parser.add_argument('--replay_modulo', type=int, help="Do experience replay session once every X frames.")
    args = parser.parse_args()

    # Defaults:
    epochs              = 10000
    memory              = 20000
    replay_batch_size   = 32
    replay_modulo       = 1

    if args.epochs:
        epochs = args.epochs
    if args.memory:
        memory = args.memory
    if args.replay_batch_size:
        replay_batch_size = args.replay_batch_size
    if args.replay_modulo:
        replay_modulo = args.replay_modulo

    if args.t:
        # training
        if args.t == 'dqn':
            deepQ = dqn.DQN(dueling=args.duel==True, use_double=args.double==True, epochs=epochs, memory=memory, replay_batch_size=replay_batch_size, replay_modulo=replay_modulo)

            deepQ.run_experiment()
        elif args.t == 'a3c':
            # Asynchronous advantage actor critic
            aaac = a3c.MasterAgent()
            aaac.train()
        elif args.t == 'pdqn':
            pass
            # prioroty dqn training here (is that what its called?)
    elif args.v:
        pass
        # testing
    elif not args.t and not args.v:
        print("Please specify if you wish to perform training or validation and provide a valid algorithm to train or test with. Valid algorithms:")
        for i in valid_alg:
            print(i)


main()