import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # used by querying
    parser.add_argument(
        '--train-size', type=int, default=2000, help='training data size')
    parser.add_argument(
        '--test-size', type=int, default=2000, help='test data size')
    parser.add_argument(
        '--valid-size', type=int, default=200, help='validation data size')

    parser.add_argument(
        '--no_cuda', action='store_true', default=False, help='do not use cuda')

    parser.add_argument(
        '--query-times', type=int, default=1, help='the number of queries')
    parser.add_argument(
        '--sigma', type=float, default=1e-4, help='likelihood noise')

    parser.add_argument(
        '--num-runs', type=int, help='number of runs for UCT')

    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--seed-range', nargs='+', type=int, help='random seed range')

    parser.add_argument(
        '--dry', action='store_true', default=False, help='dry run')

    parser.add_argument(
        '--debug', action='store_true', default=False, help='debug mode')

    parser.add_argument(
        '--k', type=int, default=1, help='evaluate top k topos'
    )
    parser.add_argument(
        '--output', type=str, default='result', help='output json file name'
    )

    parser.add_argument(
        '--model', type=str, default='gnn', choices=['simulator', 'transformer', 'gp'], help='surrogate model'
    )
    parser.add_argument(
        '--traj', nargs='+', type=int, default=[64], help='trajectory numbers'
    )

    args = parser.parse_args()

    return args
