"""
Start the DSCI algorithm HTTP server for real testbed rounds.

Example:
    python -m Scripts.Exp1_Testbed.run_algo_server --host 0.0.0.0 --port 8080
    python -m Scripts.Exp1_Testbed.run_algo_server --checkpoint Results/Optimize/DSCI/...
"""

import argparse

from Src.Algorithm.Interface.algo_service import AlgoService, AlgoServiceConfig
from Src.Algorithm.Interface.api_server import run_server
from Src.Configs.testbed_config import DEFAULT as TESTBED_CFG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSCI testbed algorithm HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=TESTBED_CFG.algo_server_port,
        help="Bind port (default from testbed_config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PPO checkpoint (.pth)",
    )
    parser.add_argument(
        "--enable-training",
        action="store_true",
        help="Record transitions and run PPO updates from measurements",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of greedy inference",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Rollout steps before PPO update (default: algo_config.buffer_size)",
    )
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AlgoServiceConfig(
        checkpoint_path=args.checkpoint,
        enable_training=args.enable_training,
        deterministic=not args.stochastic,
    )
    if args.buffer_size is not None:
        cfg.buffer_size = args.buffer_size

    service = AlgoService(config=cfg)
    print(
        f"[Algo Server] http://{args.host}:{args.port}  "
        f"checkpoint={args.checkpoint!r}  training={args.enable_training}"
    )
    run_server(host=args.host, port=args.port, service=service, debug=args.debug)


if __name__ == "__main__":
    main()
