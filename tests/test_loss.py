import torch

from upt.loss import interaction_loss, pair_labels


def output(logit):
    boxes = torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])
    return dict(
        boxes=boxes, h_idx=torch.tensor([0, 0]), o_idx=torch.tensor([1, 2]),
        logits=torch.full((2, 4), logit, requires_grad=True),
        prior=torch.tensor([[0.9, 0.9, 0.0, 0.0], [0.8, 0.0, 0.8, 0.0]]),
    )


TARGET = dict(boxes_h=torch.tensor([[0.0, 0, 10, 10]]), boxes_o=torch.tensor([[20.0, 20, 30, 30]]), labels=torch.tensor([1]))


def test_pair_labels():
    assert pair_labels(output(0.0), TARGET, 4).tolist() == [[0, 1, 0, 0], [0, 0, 0, 0]]
    empty = dict(boxes_h=torch.zeros(0, 4), boxes_o=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long))
    assert pair_labels(output(0.0), empty, 4).sum() == 0


def test_loss_counts_positives_and_prefers_correct_logits():
    loss, positives = interaction_loss([output(0.0)], [TARGET])
    assert positives == 1 and loss.isfinite() and loss > 0
    out = output(0.0)
    out["logits"] = torch.tensor([[-9.0, 9, 0, 0], [-9, 0, -9, 0]])  # right answer; zero-prior entries are ignored
    assert interaction_loss([out], [TARGET])[0] < loss


def test_gradient_only_where_prior_is_nonzero():
    out = output(0.0)
    interaction_loss([out], [TARGET])[0].backward()
    assert ((out["logits"].grad != 0) == (out["prior"] > 0)).all()
