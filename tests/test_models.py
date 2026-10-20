import pytest
import torch

from aead.config import Config
from aead.engine import build_detector


@pytest.mark.parametrize("name", ["conv_ae", "vae", "memae"])
def test_forward_backward_and_shapes(name):
    cfg = Config.from_dict({"data": {"image_size": 64},
                            "model": {"name": name, "channels": [8, 16, 32, 32],
                                      "latent_dim": 16}})
    detector = build_detector(cfg)
    x = torch.rand(2, 3, 64, 64)

    out = detector.autoencoder(x)
    assert out["recon"].shape == x.shape
    assert out["recon"].min() >= 0 and out["recon"].max() <= 1  # sigmoid output

    losses = detector.autoencoder.loss(x, out, ssim_weight=0.5)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in detector.autoencoder.parameters())

    detector.eval()
    amap, score, recon = detector(x)
    assert amap.shape == (2, 1, 64, 64) and score.shape == (2,) and recon.shape == x.shape


def test_vae_is_deterministic_in_eval():
    cfg = Config.from_dict({"data": {"image_size": 64},
                            "model": {"name": "vae", "channels": [8, 16, 32, 32]}})
    detector = build_detector(cfg).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(detector(x)[1], detector(x)[1])


def test_memae_attention_is_sparse_and_normalised():
    cfg = Config.from_dict({"data": {"image_size": 64},
                            "model": {"name": "memae", "channels": [8, 16, 32, 32],
                                      "memory_size": 20, "shrink_threshold": 0.05}})
    model = build_detector(cfg).autoencoder.eval()
    with torch.no_grad():
        w = model(torch.rand(4, 3, 64, 64))["attention"]
    assert w.shape == (4 * 4 * 4, 20)  # one query per spatial position of the 4x4 map
    assert torch.allclose(w.sum(dim=1), torch.ones(len(w)), atol=1e-4)
    assert (w == 0).any()  # hard shrinkage zeroes out weak memory slots


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        Config.from_dict({"data": {"image_size": 100}})  # not divisible by 16
    with pytest.raises(KeyError):
        Config.from_dict({"train": {"epoch": 3}})  # typo in key
