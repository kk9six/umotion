if __name__ == "__main__":
    import os, sys, torch
    import os.path as osp
    import numpy as np
    from tqdm import tqdm

    current = osp.realpath(__file__)
    base_dir = osp.dirname(osp.dirname(osp.dirname(current)))
    os.chdir(base_dir)
    sys.path.append(osp.join(base_dir, "src"))

    from loguru import logger

    train_data = torch.load("datasets/raw/uip/train.pt")
    n_train = len(train_data["fnames"])
    test_data = torch.load("datasets/raw/uip/test.pt")
    n_test = len(test_data["fnames"])

    uwb, uwb_gt = train_data["vuwb"], train_data["uwb_gt"]
    triu_indices = torch.triu_indices(6, 6, offset=1)
    uwb = torch.vstack(
        [uwb[i][:, triu_indices[0], triu_indices[1]] for i in range(n_train)]
    )
    uwb_gt = torch.vstack(
        [uwb_gt[i][:, triu_indices[0], triu_indices[1]] for i in range(n_train)]
    )

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import RANSACRegressor

    logger.info("Start training predictors")
    predictors = []
    for i in range(15):
        predictor = make_pipeline(
            PolynomialFeatures(degree=2), RANSACRegressor(random_state=3796)
        )
        predictor.fit(uwb[:, i].reshape(-1, 1), uwb_gt[:, i])
        predictors.append(predictor)

    from config import masks
    from data.dataset import Motion
    from utils.smpl import get_smpl_model

    triu_indices = torch.triu_indices(6, 6, offset=1)
    os.makedirs("datasets/processed/uip", exist_ok=True)
    for label, data in zip(["train", "test"], [train_data, test_data]):
        logger.info(f"Start processing {label} data")
        processed_data = []
        for i in tqdm(range(len(data["pose"]))):
            dists = data["vuwb"][i]
            dists = dists[:, triu_indices[0], triu_indices[1]]
            dists = torch.tensor(
                np.array(
                    [predictors[i].predict(dists[:, i].reshape(-1, 1)) for i in range(15)]
                ).T
            ).float()
            if label == "train":
                dists_gt = data["uwb_gt"][i][:, triu_indices[0], triu_indices[1]]
            else:
                smpl = get_smpl_model("neutral")
                motion = Motion(
                    smpl,
                    data["pose"][i],
                    "axis_angle",
                    data["beta"][i],
                    data["tran"][i],
                )
                dists_gt = (
                    motion.vertices[:, masks.amass_vertices][:, masks.y_idxs]
                    - motion.vertices[:, masks.amass_vertices][:, masks.x_idxs]
                ).norm(dim=-1)
            processed_data.append(
                {
                    "dists": dists,
                    "dists_gt": dists_gt,
                    "betas": data["beta"][i],
                    "poses": data["pose"][i],
                    "accs": data["acc"][i],
                    "oris": data["ori"][i],
                    "trans": data["tran"][i],
                    "gender": "neutral",
                }
            )
        torch.save(processed_data, f"datasets/processed/uip/{label}.pt")
