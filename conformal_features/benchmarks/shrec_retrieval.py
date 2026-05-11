"""SHREC shape retrieval benchmark using conformal features.

Usage:
    python -m conformal_features.benchmarks.shrec_retrieval \
        --data_dir /path/to/shrec \
        --features conformal \
        --k 10

Feature sets:
    A: xyz + normals (6D)
    B: HKS (16D) — isometry baseline
    C: conformal only (7D)
    D: conformal + isometry (10D)

Computes per-shape global descriptors by averaging per-vertex features,
then evaluates nearest-neighbor retrieval using cosine similarity.
Reports mean Average Precision (mAP).
"""
import argparse
import torch
import torch.nn.functional as F


def shape_descriptor(vertices, faces, feature_set):
    """Compute a global shape descriptor by averaging per-vertex features."""
    from conformal_features.benchmarks.shapenet_classify import extract_features
    feats = extract_features(vertices, faces, feature_set)
    return feats.mean(dim=0)


def compute_map(descriptors, labels, k=10):
    """Compute mean Average Precision for retrieval.

    Args:
        descriptors: (N, D) Tensor — one descriptor per shape
        labels: (N,) LongTensor — class labels
        k: number of retrievals

    Returns:
        mAP: float
    """
    N = descriptors.shape[0]
    sims = F.cosine_similarity(descriptors.unsqueeze(1), descriptors.unsqueeze(0), dim=2)

    aps = []
    for i in range(N):
        sim_i = sims[i].clone()
        sim_i[i] = -float('inf')  # exclude self
        _, indices = sim_i.topk(k)
        retrieved_labels = labels[indices]
        relevant = (retrieved_labels == labels[i]).float()

        if relevant.sum() == 0:
            continue

        precision_at_k = torch.cumsum(relevant, dim=0) / torch.arange(1, k + 1, dtype=torch.float32)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap.item())

    return sum(aps) / len(aps) if aps else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--features', type=str, default='conformal',
                        choices=['xyz', 'hks', 'conformal', 'conformal+iso'])
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    print(f"Running SHREC retrieval with features={args.features}, k={args.k}")
    print(f"Data dir: {args.data_dir}")
    print("TODO: integrate SHREC dataloader (load .off meshes, extract features, compute mAP)")


if __name__ == '__main__':
    main()
