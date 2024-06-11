import os
from prefect import flow, get_run_logger
from typing import Dict, Any
from jobs.deploy import put_model_to_service, deploy_prefect_flow, create_or_update_prefect_vars
import hydra
from omegaconf import DictConfig

PREFECT_MONITOR_WORK_POOL = os.getenv('PREFECT_MONITOR_WORK_POOL', 'production-model-pool')

@flow(name='deploy_flow')
def deploy_flow(cfg: Dict[str, Any],metadata_file_name: str):
    deploy_cfg = cfg.deploy
    # save_dir에서 모델을 설정하도록 서비스를 트리거
    put_model_to_service(put_model_to_service)

    prefect_kv_vars = {
        "current_model_metadata_file": metadata_file_name,
        "monitor_pool_name": PREFECT_MONITOR_WORK_POOL
    }
    # prefect vars 생성 or 업데이트
    create_or_update_prefect_vars(prefect_kv_vars)

    
    deploy_prefect_flow(deploy_cfg.prefect.work_root,
                        deploy_cfg.prefect.deployment_name)

def start(cfg: DictConfig):
    deploy=cfg.deploy
    deploy_flow(cfg,deploy.model_name,deploy.model_metadata_file_path)
   
if __name__ == "__main__":
    start()
  