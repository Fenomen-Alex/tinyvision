"""Regression tests for deterministic preprocessing parity.

Invariant: the normalisation used at inference must be identical to the one
used to build the validation transform, including when custom statistics
(``normalize_mean``/``normalize_std``) are supplied. The score path must read
those statistics from the checkpoint rather than assuming ImageNet.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from tv.config import ExperimentConfig, load_config, save_config
from tv.data.loaders import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _build_transforms,
    normalize_stats,
)
from tv.models.factory import build_classifier
from tv.reliability import Sample, _eval_transform, score_framework


def test_normalize_stats_default_is_imagenet():
    mean, std = normalize_stats({})
    assert mean == IMAGENET_MEAN
    assert std == IMAGENET_STD


def test_normalize_stats_custom_takes_precedence():
    mean, std = [0.5, 0.4, 0.3], [0.2, 0.2, 0.2]
    got_mean, got_std = normalize_stats({"normalize_mean": mean, "normalize_std": std})
    assert got_mean == mean
    assert got_std == std


def _img(seed: int, h: int = 230, w: int = 310) -> Image.Image:
    arr = (np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_eval_transform_matches_training_normalization():
    """The inference transform equals the deterministic (is_train=False) transform."""
    mean, std = [0.5, 0.4, 0.3], [0.2, 0.22, 0.18]
    aug = {"normalize_mean": mean, "normalize_std": std}
    train_eval = _build_transforms(224, aug, is_train=False)
    infer = _eval_transform(224, mean, std)
    img = _img(0)
    assert torch.equal(train_eval(img), infer(img))


def test_score_framework_honors_checkpoint_normalization(tmp_path):
    """score_framework must normalise with the checkpoint's stored statistics."""
    mean, std = [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]
    model = build_classifier("shufflenet_v2_x1_0", 2, pretrained=False)
    torch.manual_seed(0)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config_model": "shufflenet_v2_x1_0",
        "num_classes": 2,
        "normalize_mean": mean,
        "normalize_std": std,
    }
    cpath = tmp_path / "ckpt.pt"
    torch.save(ckpt, cpath)

    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"
        _img(i).save(p)
        paths.append(str(p))
    samples = [
        Sample(image_path=p, label=0, source_id="s0", environment_id="e0")
        for p in paths
    ]

    scores = score_framework(cpath, samples, image_size=224, batch_size=2)

    def manual(tfm):
        model.eval()
        xs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
        with torch.no_grad():
            probs = torch.softmax(model(xs), dim=1).numpy()
        return np.array([row[0] for row in probs])

    assert np.allclose(scores, manual(_eval_transform(224, mean, std)), atol=1e-5)
    # Sanity: the two normalisations genuinely differ on the same probe image,
    # so the equality above is not vacuous.
    probe = _img(7)
    assert not torch.allclose(
        _eval_transform(224, mean, std)(probe), _eval_transform(224)(probe)
    )


def test_augmentation_config_roundtrip(tmp_path):
    cfg = ExperimentConfig(model="resnet18", experiment_id="x")
    cfg.train.augmentation.normalize_mean = [0.1, 0.2, 0.3]
    cfg.train.augmentation.normalize_std = [0.4, 0.5, 0.6]
    cfg.train.augmentation.noise_std = 0.02
    cfg.train.augmentation.blur_prob = 0.3
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.train.augmentation.normalize_mean == [0.1, 0.2, 0.3]
    assert loaded.train.augmentation.normalize_std == [0.4, 0.5, 0.6]
    assert loaded.train.augmentation.noise_std == 0.02
    assert loaded.train.augmentation.blur_prob == 0.3
