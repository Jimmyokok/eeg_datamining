import argparse


def build_args():
    parser = argparse.ArgumentParser(description="K-means")
    parser.add_argument("--data", type=str, default='IV-I')
    parser.add_argument("--id", type=str, default='d')
    parser.add_argument("--model", type=str, default='svm')
    parser.add_argument("--trial_begin", type=float, default=0.5)
    parser.add_argument("--trial_end", type=float, default=2.5)
    parser.add_argument("--lo", type=int, default=8)
    parser.add_argument("--hi", type=int, default=15)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--valid", type=str, default='cross')
    parser.add_argument("--ratio", type=float, default=0.7)
    args = parser.parse_args()
    return args