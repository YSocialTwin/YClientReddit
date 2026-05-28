from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def prepare_run_dir(run_dir: Path, prompts_path: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompts_path, run_dir / "prompts.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", default="admin")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--content-recsys", default="ReverseChronoFollowersPopularity")
    parser.add_argument("--follow-recsys", default="PreferentialAttachment")
    args = parser.parse_args()

    client_root = Path(args.client_root).resolve()
    config_path = Path(args.config).resolve()
    prompts_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    prepare_run_dir(output_dir, prompts_path)
    config = load_json(config_path)

    sys.path.insert(0, str(client_root))
    clients = importlib.import_module("y_client.clients")
    recsys = importlib.import_module("y_client.recsys")

    client_name = config["simulation"]["client"]
    if hasattr(clients, client_name):
        client_cls = getattr(clients, client_name)
    elif client_name == "YClientBase":
        client_cls = getattr(importlib.import_module("y_client.clients.client_base"), client_name)
    elif client_name == "YClientWeb":
        client_cls = getattr(importlib.import_module("y_client.clients.client_web"), client_name)
    else:
        raise AttributeError(f"Unsupported client class: {client_name}")
    content_recsys = getattr(recsys, args.content_recsys)()
    follow_recsys = getattr(recsys, args.follow_recsys)(leaning_bias=1.5)
    output_agents = output_dir / f"{config['simulation']['name']}_agents.json"

    init_params = inspect.signature(client_cls.__init__).parameters
    if "data_base_path" in init_params:
        experiment = client_cls(
            config,
            f"{output_dir}{os.sep}",
            agents_output=str(output_agents),
            owner=args.owner,
            first_run=True,
        )
    else:
        experiment = client_cls(
            str(config_path),
            str(prompts_path),
            agents_output=str(output_agents),
            owner=args.owner,
        )

    experiment.set_recsys(content_recsys, follow_recsys)
    experiment.create_initial_population()
    experiment.save_agents()
    experiment.run_simulation()
    print(json.dumps({"agents_file": str(output_agents), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
