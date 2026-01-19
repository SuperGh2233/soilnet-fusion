import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from train_ml_baselines import train_ml_baselines


def resolve_oc_max(dataset_name: str) -> float:
    name = dataset_name.lower()
    if name == "china":
        return 30.8
    if name == "lucas":
        return 87.0
    if name == "raca":
        return 4115.0
    raise ValueError(f"未知数据集 {dataset_name}，无法确定 OC_MAX。")


def create_transforms(use_srtm: bool, oc_max: float):
    from dataset.dataset_loader_china import myNormalize, myToTensor, Augmentations

    if use_srtm:
        img_norm = [
            [(0, 7), (0, 1)],
            [(7, 12), (-1, 1)],
            [(12), (-4, 2963)],
            [(13), (0, 90)],
        ]
    else:
        img_norm = [
            [(0, 7), (0, 1)],
            [(7, 12), (-1, 1)],
        ]

    mynorm = myNormalize(img_bands_min_max=img_norm, oc_min=0, oc_max=oc_max)
    my_to_tensor = myToTensor()
    my_aug = Augmentations()
    train_transform = transforms.Compose([mynorm, my_to_tensor, my_aug])
    eval_transform = transforms.Compose([mynorm, my_to_tensor])
    return train_transform, eval_transform


def build_china_datasets(use_static=True, use_srtm=True, oc_max=30.8):
    from dataset.dataset_loader_china import ChinaSNDatasetClimate

    bands = getattr(config, "bands", list(range(12)))
    train_tf, eval_tf = create_transforms(use_srtm, oc_max)

    static_csv_path = getattr(config, "static_csv_path", None)
    use_static = use_static and static_csv_path not in (None, "", "None")

    dataset_cls = ChinaSNDatasetClimate
    using_static = False
    if use_static:
        try:
            from dataset.dataset_loader_china_static import ChinaSNDatasetClimateStatic

            dataset_cls = ChinaSNDatasetClimateStatic
            using_static = True
        except ImportError:
            print("[WARN] 未找到 ChinaSNDatasetClimateStatic，改用无静态特征的数据集。")
            using_static = False

    def make_dataset(root_dir, tfm):
        kwargs = dict(
            l8_dir=root_dir,
            csv_dir=config.lucas_csv_path,
            climate_csv_folder=config.climate_csv_folder_path,
            l8_bands=bands,
            transform=tfm,
        )
        if using_static:
            kwargs["static_csv_path"] = static_csv_path
        return dataset_cls(**kwargs)

    train_ds = make_dataset(config.train_l8_folder_path, train_tf)
    val_ds = make_dataset(config.val_l8_folder_path, eval_tf)
    test_ds = make_dataset(config.test_l8_folder_path, eval_tf)

    static_dim = 0
    lulc_classes = 0
    if using_static:
        static_dim = getattr(train_ds, "get_static_numeric_dim", lambda: 0)()
        if static_dim == 0:
            static_dim = getattr(train_ds, "get_static_feature_dim", lambda: 0)()
        lulc_classes = getattr(train_ds, "get_lulc_num_classes", lambda: 0)()

    return train_ds, val_ds, test_ds, using_static, static_dim, lulc_classes


def to_one_hot(indices: torch.Tensor, num_classes: int):
    if num_classes <= 0:
        return torch.zeros(indices.shape[0], 0, device=indices.device)
    idx = indices.clone().long()
    mask = idx < 0
    idx = idx.clamp(min=0)
    out = torch.zeros(idx.shape[0], num_classes, device=indices.device)
    out.scatter_(1, idx.unsqueeze(1), 1.0)
    out[mask] = 0
    return out


def collect_modalities(dataset, batch_size, num_workers, expect_static, lulc_classes):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    imgs, ts_feats, static_feats, labels = [], [], [], []
    for batch in loader:
        features, target = batch
        if expect_static:
            if not isinstance(features, (list, tuple)) or len(features) < 4:
                raise ValueError("数据集中未包含静态分支，但设置了 expect_static=True。")
            img, ts, static_vec, lulc_idx = features[:4]
            static_vec = static_vec.float()
            one_hot = to_one_hot(lulc_idx, lulc_classes)
            static_vec = torch.cat([static_vec, one_hot], dim=1)
        else:
            if not isinstance(features, (list, tuple)) or len(features) < 2:
                raise ValueError("数据返回格式不符合预期。")
            img, ts = features[:2]
            static_vec = torch.zeros(img.shape[0], 0)

        imgs.append(img.float())
        ts_feats.append(ts.float())
        static_feats.append(static_vec.float())
        labels.append(target.float())

    img_arr = torch.cat(imgs, dim=0).numpy()
    ts_arr = torch.cat(ts_feats, dim=0).numpy()
    if static_feats and static_feats[0].numel() > 0:
        static_arr = torch.cat(static_feats, dim=0).numpy()
    else:
        static_arr = np.zeros((img_arr.shape[0], 0), dtype=np.float32)
    label_arr = torch.cat(labels, dim=0).numpy()
    return img_arr, ts_arr, static_arr, label_arr


def main():
    parser = argparse.ArgumentParser(description="自动运行 RF / XGB 基线")
    parser.add_argument("-d", "--dataset", type=str, default="CHINA")
    parser.add_argument("--use_static", action="store_true", default=False)
    parser.add_argument("--use_srtm", action="store_true", default=True)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-nw", "--num_workers", type=int, default=8)
    parser.add_argument(
        "--save_prefix",
        type=str,
        default=os.path.join("checkpoints", "ml_baseline"),
    )
    args = parser.parse_args()

    if args.dataset.upper() != "CHINA":
        raise NotImplementedError("当前脚本仅支持 CHINA 数据集。")

    oc_max = resolve_oc_max(args.dataset)

    train_ds, val_ds, test_ds, using_static, _, lulc_classes = build_china_datasets(
        use_static=args.use_static, use_srtm=args.use_srtm, oc_max=oc_max
    )

    print("收集训练特征 ...")
    train_img, train_ts, train_static, train_y = collect_modalities(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        expect_static=using_static,
        lulc_classes=lulc_classes,
    )

    print("收集验证特征 ...")
    val_img, val_ts, val_static, val_y = collect_modalities(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        expect_static=using_static,
        lulc_classes=lulc_classes,
    )

    os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)

    results = train_ml_baselines(
        X_train_list=[train_img, train_ts, train_static],
        y_train=train_y,
        X_val_test_list=[val_img, val_ts, val_static],
        y_val_test=val_y,
        save_path_prefix=args.save_prefix,
        oc_max=oc_max,
    )

    print("最终结果（已反归一化）：")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"R2={metrics['R2']:.4f}, RPIQ={metrics['RPIQ']:.4f}, "
            f"MEC={metrics['MEC']:.4f}, CCC={metrics['CCC']:.4f}"
        )


if __name__ == "__main__":
    main()

