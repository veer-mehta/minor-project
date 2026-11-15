"""UPT: a frozen DETR detector followed by the unary-pairwise interaction head."""
import torch
from torch import nn

from .detector import build_detr, detect, select_proposals
from .layers import CompetitiveLayer, CooperativeLayer, Fusion, pair_spatial_features

HUMAN = 0  # class index of "person" in both fine-tuned detectors


def object_to_target(meta, name):
    """For each detector object class, the list of interaction classes it can take part in."""
    if name == "hicodet":
        valid = [[] for _ in meta["objects"]]
        for _, obj, verb in meta["correspondence"]:
            valid[obj].append(verb)
        return valid
    # V-COCO annotations count objects from 1 (0 is background); the detector counts from 0
    return [[a for a, objs in enumerate(meta["action_to_object"]) if o in objs] for o in range(1, 81)]


class InteractionHead(nn.Module):
    def __init__(self, num_classes, object_to_target, token_size=256, pair_size=512, feature_channels=2048):
        super().__init__()
        self.box_pair_predictor = nn.Linear(pair_size * 2, num_classes)
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, pair_size), nn.ReLU()
        )
        self.coop_layer = nn.ModuleDict(
            {"mod_enc": nn.ModuleList(CooperativeLayer(token_size, pair_size) for _ in range(2))}
        )
        self.comp_layer = CompetitiveLayer(pair_size * 2)
        self.mbf = Fusion(token_size * 2, pair_size, pair_size)
        self.mbf_g = Fusion(feature_channels, pair_size, pair_size)

        valid = torch.zeros(len(object_to_target), num_classes)
        for obj, targets in enumerate(object_to_target):
            valid[obj, targets] = 1
        # Not saved: the released checkpoints do not contain it
        self.register_buffer("valid", valid, persistent=False)

    def forward(self, features, sizes, proposals):
        """features: (B, C, h, w) backbone map. sizes: (B, 2) as (H, W). proposals: humans first.

        Returns per image: boxes (N, 4); h_idx, o_idx (M,) indices into boxes; objects (M,);
        logits (M, K); prior (M, K), the detection confidence of the pair, zero for
        interactions the object class cannot take part in.
        """
        global_features = features.mean((2, 3))
        power = 1.0 if self.training else 2.8  # sharpen detection scores at inference
        outputs = []
        for props, size, global_feature in zip(proposals, sizes, global_features):
            boxes, scores, labels, tokens = props["boxes"], props["scores"], props["labels"], props["hidden_states"]
            n, n_h = len(boxes), int((labels == HUMAN).sum())
            i, j = torch.meshgrid(torch.arange(n, device=boxes.device), torch.arange(n, device=boxes.device), indexing="ij")
            # Pair every human with every other box
            h_idx, o_idx = torch.nonzero((i != j) & (i < n_h)).unbind(1)
            if len(h_idx) == 0:
                empty = boxes.new_zeros(0, self.box_pair_predictor.out_features)
                outputs.append(dict(boxes=boxes, h_idx=h_idx, o_idx=o_idx, objects=h_idx, logits=empty, prior=empty))
                continue

            spatial = pair_spatial_features(boxes[i.flatten()], boxes[j.flatten()], size[0], size[1])
            spatial = self.spatial_head(spatial).view(n, n, -1)
            for layer in self.coop_layer["mod_enc"]:
                tokens = layer(tokens, spatial)

            pair_spatial = spatial[h_idx, o_idx]
            pair_tokens = torch.cat([
                self.mbf(torch.cat([tokens[h_idx], tokens[o_idx]], 1), pair_spatial),
                self.mbf_g(global_feature[None], pair_spatial),
            ], 1)
            pair_tokens = self.comp_layer(pair_tokens)

            prior = (scores[h_idx] * scores[o_idx]).pow(power)[:, None] * self.valid[labels[o_idx]]
            outputs.append(dict(
                boxes=boxes, h_idx=h_idx, o_idx=o_idx, objects=labels[o_idx],
                logits=self.box_pair_predictor(pair_tokens), prior=prior,
            ))
        return outputs


class UPT(nn.Module):
    def __init__(self, num_classes, object_to_target, num_object_classes=80):
        super().__init__()
        self.detector = build_detr(num_object_classes)
        self.interaction_head = InteractionHead(num_classes, object_to_target)

    def forward(self, images):
        """images: list of normalised (3, H, W) tensors. See InteractionHead.forward for the output."""
        features, sizes, detections = detect(self.detector, images)
        return self.interaction_head(features, sizes, [select_proposals(d) for d in detections])


def to_detections(output):
    """Flatten one image's output into scored (human, object, interaction) triplets."""
    pair, label = torch.nonzero(output["prior"]).unbind(1)
    return dict(
        boxes_h=output["boxes"][output["h_idx"][pair]],
        boxes_o=output["boxes"][output["o_idx"][pair]],
        objects=output["objects"][pair],
        labels=label,
        scores=output["logits"][pair, label].sigmoid() * output["prior"][pair, label],
    )
