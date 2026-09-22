import torch

from research.arc_decision_model import ARCDecisionDynamics, decision_utility, training_loss


def test_forward_and_loss():
    model = ARCDecisionDynamics(width=32, depth=2, cond_dim=48)
    b, h, w = 4, 13, 17
    cur = torch.randint(0, 16, (b, h, w))
    prev = torch.randint(0, 16, (b, h, w))
    action = torch.tensor([1, 2, 6, 7])
    click_x = torch.tensor([0, 0, 8, 0])
    click_y = torch.tensor([0, 0, 5, 0])
    nxt = torch.randint(0, 16, (b, h, w))
    changed = torch.tensor([1, 1, 0, 1])
    progressed = torch.tensor([0, 1, 0, 0])
    died = torch.tensor([0, 0, 1, 0])

    out = model(cur, prev, action, click_x, click_y)
    assert out.next_grid_logits.shape == (b, 16, h, w)
    loss, parts = training_loss(out, nxt, changed, progressed, died)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(parts) == {"grid_ce", "change_bce", "progress_bce", "death_bce"}
    utility = decision_utility(out)
    assert utility.shape == (b,)
    assert torch.isfinite(utility).all()
