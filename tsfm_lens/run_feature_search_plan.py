"""CPU-only register/screen/plan/analyze workflow; never launches models.

Catalog JSON declares primary_estimand, alpha, minimum_absolute_mean for the
four production endpoints, and cases=[{id,prepared,report,model_order}].
Optional predictions maps model names to exact expected NPZ paths. Register
before screen reads results; screen accepts no new thresholds. Plan emits
configs and prepare/infer commands for the existing feature-interference CLI.
Every operation requires a new output directory.
"""
import argparse
from tsfm_lens.experiments.feature_search.replication_planner import register, screen, plan, analyze_replication


def main(argv=None):
    """Keep thresholds at registration and fresh seeds at plan creation only."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    registration = commands.add_parser('register')
    registration.add_argument('--catalog', required=True)
    registration.add_argument('--out', required=True)
    screening = commands.add_parser('screen')
    screening.add_argument('--registration', required=True)
    screening.add_argument('--out', required=True)
    planning = commands.add_parser('plan')
    planning.add_argument('--screen', required=True)
    planning.add_argument('--out', required=True)
    planning.add_argument('--fresh-seed-base', type=int)
    planning.add_argument('--n-bundles', type=int)
    planning.add_argument('--device', default='cpu')
    replication = commands.add_parser('analyze-replication')
    replication.add_argument('--plan', required=True)
    replication.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'register':
        return register(args.catalog, args.out)
    if args.command == 'screen':
        return screen(args.registration, args.out)
    if args.command == 'plan':
        return plan(args.screen, args.out, fresh_seed_base=args.fresh_seed_base, n_bundles=args.n_bundles, device=args.device)
    return analyze_replication(args.plan, args.out)


if __name__ == '__main__':
    main()
