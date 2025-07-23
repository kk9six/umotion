import os, datetime
import os.path as osp
from omegaconf import OmegaConf

def do_train(config_name="config.yaml"):
    base_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    os.chdir(base_dir)
    cfg = OmegaConf.load(osp.join(base_dir, f"configs/{config_name}"))
    OmegaConf.update(cfg, "dirs.base", base_dir)
    if "pose_estimator" not in cfg.dirs:
        OmegaConf.update(cfg, "dirs.pose_estimator", osp.join(cfg.dirs.models, f"pose-estimator-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))
    if osp.exists(cfg.dirs.pose_estimator) and not cfg.training.overwrite:
        raise FileExistsError(
            "Path already exists, delete it / choose another path / set --overwrite"
        )
    OmegaConf.update(cfg, "dirs.checkpoints", osp.join(cfg.dirs.pose_estimator, "checkpoints"))
    os.makedirs(cfg.dirs.pose_estimator, exist_ok=True)
    os.makedirs(cfg.dirs.checkpoints, exist_ok=True)
    OmegaConf.save(cfg, osp.join(cfg.dirs.pose_estimator, "config.yaml"))

    from network.train import PoseEstimatorTrainer

    PoseEstimatorTrainer.train(cfg)
    return cfg.dirs.pose_estimator

def do_test(model_name, shape_estimator_name, checkpoint=None):
    base_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    os.chdir(base_dir)
    cfg = OmegaConf.load(osp.join(base_dir, "models", model_name, "config.yaml"))
    from network.test import PoseEstimatorTester
    PoseEstimatorTester.test(cfg.dirs.pose_estimator, shape_estimator_name, checkpoint)

if __name__ == "__main__":
    # model_name = do_train("config.yaml") # no noise
    # model_name = do_train("config_noise.yaml") # noisy distance

    model_name = "pose_estimator"
    shape_estimator_name = "shape_estimator"
    do_test(model_name, shape_estimator_name)
