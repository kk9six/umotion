import os, datetime
import os.path as osp
from omegaconf import OmegaConf

def do_train():
    base_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    os.chdir(base_dir)
    cfg = OmegaConf.load(osp.join(base_dir, "configs/config.yaml"))
    OmegaConf.update(cfg, "dirs.base", base_dir)
    if "shape_estimator" not in cfg.dirs:
        OmegaConf.update(cfg, "dirs.shape_estimator", osp.join(cfg.dirs.models, f"shape-estimator-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))
    if osp.exists(cfg.dirs.shape_estimator) and not cfg.training.overwrite:
        raise FileExistsError(
            "Path already exists, delete it / choose another path / set --overwrite"
        )
    os.makedirs(cfg.dirs.shape_estimator, exist_ok=True)
    OmegaConf.save(cfg, osp.join(cfg.dirs.shape_estimator, "config.yaml"))

    from network.train import ShapeEstimatorTrainer
    ShapeEstimatorTrainer.train(cfg)

    return cfg.dirs.shape_estimator

def do_test(model_name):
    base_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    os.chdir(base_dir)
    cfg = OmegaConf.load(osp.join(base_dir, "models", model_name, "config.yaml"))
    OmegaConf.update(cfg, "dirs.base", base_dir)
    from network.test import ShapeEstimatorTester
    ShapeEstimatorTester.test(cfg.dirs.shape_estimator)

if __name__ == "__main__":
    # model_name = do_train()

    model_name = "shape_estimator"
    do_test(model_name)
