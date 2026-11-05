import torch

from nhanes_reliable.model import FusionSurvivalNet
from nhanes_reliable.survival import N_INTERVALS


def inputs(b=4, n_tab=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    days = torch.rand(b, 7, 2, 1440, generator=g)
    mask = torch.ones(b, 7, dtype=torch.bool)
    mask[:, 5:] = False
    return days, mask, torch.randn(b, n_tab, generator=g), torch.ones(b, 4)


def net(use_wearable=True):
    torch.manual_seed(0)
    return FusionSurvivalNet(n_tab=10, use_wearable=use_wearable).eval()


def test_output_shape():
    assert net()(*inputs()).shape == (4, N_INTERVALS)


def test_masked_days_ignored():
    model, (days, mask, tab, flags) = net(), inputs()
    before = model(days, mask, tab, flags)
    days[:, 5:] = 99.0
    torch.testing.assert_close(model(days, mask, tab, flags), before)


def test_day_order_invariant():
    model, (days, mask, tab, flags) = net(), inputs()
    before = model(days, mask, tab, flags)
    order = torch.tensor([3, 0, 4, 1, 2, 5, 6])
    torch.testing.assert_close(model(days[:, order], mask[:, order], tab, flags), before, atol=1e-5, rtol=1e-5)


def test_no_valid_days_finite():
    model, (days, mask, tab, flags) = net(), inputs()
    mask[0] = False
    out = model(days, mask, tab, flags)
    assert torch.isfinite(out).all()
    flags_off = flags.clone()
    flags_off[0, 0] = 0
    torch.testing.assert_close(out[0], model(days, mask, tab, flags_off)[0])


def test_no_wearable_branch():
    model, (_, _, tab, flags) = net(use_wearable=False), inputs()
    assert model(None, None, tab, flags).shape == (4, N_INTERVALS)
