import argparse
import os
from core.parser import parse_args
from core.train_cmd import start_train
from core.test_cmd import start_test
from MicroEvoEvalCore.calculate_phy import start_eval

if __name__ == "__main__":
    args = parse_args()
    if args.use_mode == "train":
        print("Train mode")
        start_train(args)
    elif args.use_mode == "eval" or args.use_mode == "evaluate":
        print("Evaluate mode")
        start_eval(args)
    elif args.use_mode == "test":
        print("Test mode")
        start_test(args)
    
    