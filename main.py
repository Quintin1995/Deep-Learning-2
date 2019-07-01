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
    mutex1.add_argument('-t',type=str, help="Flag for training one of the methods. Takes one positional argument (being the name of the method).")
    mutex1.add_argument('-v',type=str,help="Flag for validation/testing one of the methods. Takes one positional argument.")
    parser.add_argument('--render', default=True, help="Enables rendering the game during training/testing so we can run it on peregrine.")
    parser.add_argument('--target', default=False, action='store_true', help="Enables frozen weights target network for DQN model.")
    parser.add_argument('--dueling', default=False, action='store_true',help="Uses a dueling network structure when running with a DQN network.")
    parser.add_argument('--epochs', default=5000, type=int, help="Specify amount of epochs to run.")
    parser.add_argument('--memory', default=10000, type=int, help="Specify amount of experiences we can store at once.")
    parser.add_argument('--replay_batch_size', default=32, type=int, help="Specify amount of experiences to replay per replay session.")
    parser.add_argument('--replay_modulo', default=5, type=int, help="Do experience replay session once every X frames.")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="amount of consecutive frames to grab")
    args = parser.parse_args()


    if args.t:
        # training
        if args.t == 'dqn':
            deepQ = dqn.DQN(args)

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