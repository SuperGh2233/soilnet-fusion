import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

from train_utils import evaluate_regression_metrics


def _ensure_numpy(array):
    """兼容 torch Tensor / list / numpy，统一转换成 numpy.float32。"""
    if array is None:
        return None
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def _pool_image_features(img, mode="mean"):
    """
    图像特征聚合：默认全局平均池化，也支持取中心像素。
    img: (B, C, H, W)
    """
    if img.ndim != 4:
        raise ValueError(f"期望图像 shape=(B,C,H,W)，收到 {img.shape}")
    if mode == "center":
        h_mid = img.shape[2] // 2
        w_mid = img.shape[3] // 2
        return img[:, :, h_mid, w_mid]  # (B, C)
    return img.mean(axis=(2, 3))  # (B, C)


def _pool_temporal_features(ts, stats=("mean", "std")):
    """
    时序特征聚合：默认取时间维度均值 + 标准差，可按需求扩展。
    ts: (B, T, F)
    """
    if ts.ndim != 3:
        raise ValueError(f"期望时序 shape=(B,T,F)，收到 {ts.shape}")
    feats = []
    if "mean" in stats:
        feats.append(ts.mean(axis=1))
    if "std" in stats:
        feats.append(ts.std(axis=1))
    return np.concatenate(feats, axis=1)


def _compute_metrics_dict(y_true, y_pred, oc_max=None):
    """
    计算与主训练脚本一致的指标，并在需要时对 RMSE/MAE 反归一化。
    """
    if oc_max is not None:
        y_true = y_true * oc_max
        y_pred = y_pred * oc_max
    rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)
    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "RPIQ": float(rpiq),
        "MAE": float(mae),
        "MEC": float(mec),
        "CCC": float(ccc),
    }


def train_ml_baselines(
    X_train_list,
    y_train,
    X_val_test_list,
    y_val_test,
    feature_norm=True,
    rf_params=None,
    xgb_params=None,
    save_path_prefix=None,
    img_pooling_mode="mean",
    ts_stats=("mean", "std"),
    oc_max=None,
):
    """
    训练 RF / XGB 传统基线模型。

    Args:
        X_train_list: [img_train, ts_train, static_train]
        y_train: 训练集标签
        X_val_test_list: [img_val_test, ts_val_test, static_val_test]
        y_val_test: 验证/测试集标签
        feature_norm: 是否对拼接后的特征做标准化
        rf_params/xgb_params: dict，超参
        save_path_prefix: 若不为 None，则保存模型
        img_pooling_mode: 图像池化方式（mean/center）
        ts_stats: 时序特征提取统计项
        oc_max: 若提供，则用于将预测值从 0-1 反归一化回真实 SOC 区间
    """
    print("准备机器学习特征 ...")
    img_train, ts_train, static_train = map(_ensure_numpy, X_train_list)
    img_test, ts_test, static_test = map(_ensure_numpy, X_val_test_list)
    y_train = _ensure_numpy(y_train).ravel()
    y_test = _ensure_numpy(y_val_test).ravel()

    # 图像特征
    feat_vis_train = _pool_image_features(img_train, mode=img_pooling_mode)
    feat_vis_test = _pool_image_features(img_test, mode=img_pooling_mode)

    # 时序特征
    feat_ts_train = _pool_temporal_features(ts_train, stats=ts_stats)
    feat_ts_test = _pool_temporal_features(ts_test, stats=ts_stats)

    # 静态特征
    feat_static_train = static_train
    feat_static_test = static_test

    # 缺失值填充
    imputer = SimpleImputer(strategy="mean")
    stacked_train = np.concatenate(
        [feat_vis_train, feat_ts_train, feat_static_train], axis=1
    )
    stacked_test = np.concatenate(
        [feat_vis_test, feat_ts_test, feat_static_test], axis=1
    )
    stacked_train = imputer.fit_transform(stacked_train)
    stacked_test = imputer.transform(stacked_test)

    # 特征标准化
    if feature_norm:
        scaler = StandardScaler()
        stacked_train = scaler.fit_transform(stacked_train)
        stacked_test = scaler.transform(stacked_test)
    else:
        scaler = None

    print(f"最终特征维度: {stacked_train.shape[1]}")

    results = {}

    # Random Forest
    print("[RF] 训练中 ...")
    rf_default = dict(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    if rf_params:
        rf_default.update(rf_params)
    rf = RandomForestRegressor(**rf_default)
    rf.fit(stacked_train, y_train)
    y_pred_rf = rf.predict(stacked_test)
    results["RF"] = _compute_metrics_dict(y_test, y_pred_rf, oc_max=oc_max)
    print(
        "[RF] RMSE={:.4f}, MAE={:.4f}, R2={:.4f}".format(
            results["RF"]["RMSE"], results["RF"]["MAE"], results["RF"]["R2"]
        )
    )

    # XGBoost
    print("[XGB] 训练中 ...")
    xgb_default = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )
    if xgb_params:
        xgb_default.update(xgb_params)
    xgb = XGBRegressor(**xgb_default)
    xgb.fit(stacked_train, y_train)
    y_pred_xgb = xgb.predict(stacked_test)
    results["XGB"] = _compute_metrics_dict(y_test, y_pred_xgb, oc_max=oc_max)
    print(
        "[XGB] RMSE={:.4f}, MAE={:.4f}, R2={:.4f}".format(
            results["XGB"]["RMSE"], results["XGB"]["MAE"], results["XGB"]["R2"]
        )
    )

    # 保存模型（可选）
    if save_path_prefix:
        joblib.dump(
            {
                "rf": rf,
                "xgb": xgb,
                "imputer": imputer,
                "scaler": scaler,
            },
            f"{save_path_prefix}_ml_baselines.pkl",
        )
        print(f"模型已保存到 {save_path_prefix}_ml_baselines.pkl")

    return results


if __name__ == "__main__":
    print("该脚本作为模块使用，请在其他训练脚本中调用 train_ml_baselines(...)")

