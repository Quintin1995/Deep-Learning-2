# Base imports
import argparse

import dqn.dqn as dqn
# import other frameworks here

valid_alg = [
    'dqn',
    'a3c',
    'pdqn']

def main():
    parser = argparse.ArgumentParser()
    mutex1 = parser.add_mutually_exclusive_group()
    # This means we can now either train or validate. Validate is just here so we can have single letter options because train and test start with the same letter
    mutex1.add_argument('-t',type=str,help="Flag for training one of the methods. Takes one positional argument (being the name of the method).")
    mutex1.add_argument('-v',type=str,help="Flag for validation/testing one of the methods. Takes one positional argument.")
    parser.add_argument('--norender',help="Disables rendering the game during training/testing so we can run it on peregrine.")
    parser.add_argument('--target', action='store_true', help="Enables frozen weights target network for DQN model.")
    parser.add_argument('--duel', action='store_true',help="Uses a dueling network structure when running with a DQN network.")
    args = parser.parse_args()
    if args.t:
        # training
        if args.t == 'dqn':
            deepQ = dqn.DQN(dueling=args.duel==True, use_target_network=args.target==True)

            deepQ.run_experiment()
        elif args.t == 'a3c':
            pass
            # asynchonous advantage actor critic training here
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