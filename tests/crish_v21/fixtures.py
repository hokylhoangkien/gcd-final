import torch

from crish.probe import LegalProbe, ProbeWeights
from crish.rebuild import prototype_radii
from crish.state import SupervisionState


def state_fixture():
    ids = torch.tensor([40, 7, 91, 15, 22, 66, 80, 12])
    features = torch.tensor([[0., 0., 0.], [1., 0., 0.], [.2, .1, 0.], [1.1, -.1, 0.],
                             [2.9, 2., 0.], [3.1, 2., 0.], [5., 4., 0.], [5.2, 4., 0.]])
    labels = (torch.tensor([0, 1, 0, 1, 2, 2, 3, 3]), torch.tensor([0, 0, 0, 1, 0, 1, 1, 1]))
    centers = tuple(torch.stack([features[y == g].mean(0) for g in y.unique()]) + .013 for y in labels)
    return SupervisionState(ids, features, torch.tensor([True, True] + [False] * 6),
                            torch.tensor([0, 1]), labels, centers,
                            tuple(prototype_radii(p) for p in centers),
                            tuple(torch.zeros(8, dtype=torch.bool) for _ in labels))


def backbone_views(state):
    return torch.cat([state.features, state.features + torch.tensor([.02, -.03, .04])])


def probe_fixture(model, offset=0):
    generator = torch.Generator().manual_seed(142 + offset)
    known = torch.randn(3, 2, 3, generator=generator)
    unlabeled = torch.randn(4, 2, 3, generator=generator)
    with torch.no_grad():
        teacher = model(unlabeled.flatten(0, 1)).reshape(4, 2, -1).clone()
    return LegalProbe(known_ids=torch.arange(100, 103) + offset, known_views=known,
                      known_targets=torch.tensor([0, 0, 1]),
                      unlabeled_ids=torch.arange(200, 204) + offset,
                      unlabeled_views=unlabeled, teacher_views=teacher,
                      proposal_ids=torch.arange(8),
                      legal_labels={100 + offset: 0, 101 + offset: 0, 102 + offset: 1},
                      weights=ProbeWeights(1., 1., .1, .5))
