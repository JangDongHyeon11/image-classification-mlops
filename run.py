from importlib import import_module
from omegaconf import OmegaConf;
import os
import argparse
import yaml
import json
# from ray.job_submission import JobSubmissionClient, JobStatus

# RAY_DASHBOARD_HOST = os.getenv("RAY_DASHBOARD_HOST", "ray")
# RAY_DASHBOARD_PORT = os.getenv("RAY_DASHBOARD_PORT", "8265")

# ray_job_client = JobSubmissionClient(
#     f"http://{RAY_DASHBOARD_HOST}:{RAY_DASHBOARD_PORT}"
# )

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="config file 입력")
args = parser.parse_args()
print(args.config)
with open(args.config, 'r') as f:
    if args.config.endswith(('.yml', '.yaml')):
        config = yaml.safe_load(f)
    else:
        config = json.load(f)
config = OmegaConf.create(config)
module = import_module('workflows.train')

# train_job_id = ray_job_client.submit_job(
#     entrypoint="python workflows/train.py", 
#     runtime_env={"env_vars": {"CONFIG": config}}
# )

module.start(config)