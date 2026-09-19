"""Focused checks for the GRSL revision data and fusion contracts."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.dataset_loader_china_static import ChinaSNDatasetClimateStatic
from prepare_revision_split import validate_split
from soilnet.submodules.semantic_aligned_fusion import (
    GatedFeatureFusion,
    TokenCrossAttentionFusion,
)


def test_buffer_validation():
    manifest = pd.DataFrame(
        {
            "Point_id": ["1", "2", "3"],
            "Latitude": [20.0, 30.0, 40.0],
            "Longitude": [100.0, 110.0, 120.0],
            "SOC": [10.0, 20.0, 70.0],
            "split": ["train", "val", "test"],
        }
    )
    distances = validate_split(manifest, buffer_km=50.0)
    assert min(distances.values()) > 50.0


def test_train_only_preprocessing():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_dir = root / "images"
        climate_dir = root / "climate"
        image_dir.mkdir()
        climate_dir.mkdir()
        for point_id in (1, 2, 3, 4):
            (image_dir / f"{point_id}_image.tif").touch()

        static_csv = root / "samples.csv"
        pd.DataFrame(
            {
                "Point_id": [1, 2, 3, 4],
                "Latitude": [20, 21, 22, 23],
                "Longitude": [100, 101, 102, 103],
                "Year": [2014] * 4,
                "SOC": [10, 20, 30, 80],
                "CLCD": ["crop", "forest", "crop", "urban"],
                "BD_g_cm3": [1.0, 3.0, 101.0, 201.0],
            }
        ).to_csv(static_csv, index=False)
        pd.DataFrame(
            {
                "Point_id": [1, 2, 3, 4],
                "20110101": [0.0, 10.0, 100.0, 200.0],
                "20110201": [2.0, 12.0, 102.0, 202.0],
            }
        ).to_csv(climate_dir / "temperature.csv", index=False)

        common = dict(
            l8_dir=str(image_dir),
            csv_dir=str(static_csv),
            climate_csv_folder=str(climate_dir),
            static_csv_path=str(static_csv),
            transform=None,
        )
        train = ChinaSNDatasetClimateStatic(
            **common,
            point_ids={"1", "2"},
            fit_climate_stats=True,
            fit_static_stats=True,
        )
        test = ChinaSNDatasetClimateStatic(
            **common,
            point_ids={"3", "4"},
            climate_stats=train.get_climate_stats(),
            static_stats=train.get_static_stats(),
        )
        assert train.get_climate_stats() == test.get_climate_stats()
        assert train.get_static_stats() == test.get_static_stats()
        assert train.get_static_stats()["mean"]["BD_g_cm3"] == 2.0
        assert train.get_static_stats()["lulc_categories"] == ["crop", "forest"]
        assert test.lulc_indices["4"] == -1
        assert test.clim_dfs[0].loc[2, "20110101"] > 1.0


def test_fusion_baselines():
    climate = torch.randn(4, 128)
    visual_vector = torch.randn(4, 384)
    visual_tokens = torch.randn(4, 16, 64)
    static = torch.randn(4, 128)
    gated = GatedFeatureFusion(128, 384, 128)
    attention = TokenCrossAttentionFusion(128, 64, 128)
    assert gated(climate, visual_vector, static).shape == (4, 128)
    assert attention(climate, visual_tokens, static).shape == (4, 128)


if __name__ == "__main__":
    test_buffer_validation()
    test_train_only_preprocessing()
    test_fusion_baselines()
    print("PASS: revision protocol checks")
