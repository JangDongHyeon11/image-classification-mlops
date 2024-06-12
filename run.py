import hydra
from importlib import import_module
from omegaconf import DictConfig,OmegaConf;


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.mode
    if mode == "train":
        module = import_module('workflows.train')
    elif mode == "eval":
        module = import_module('workflows.eval')
    elif mode == "deploy":
        module = import_module('workflows.deploy')
        
    mode_cfg = OmegaConf.merge({"dataset":cfg.dataset,"model":cfg.model, f"{mode}":cfg[mode]})  # Merge relevant configurations
    module.start(mode_cfg)
    
if __name__ == "__main__":
    main()