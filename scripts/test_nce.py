"""
Test NCE loss output
"""

from class_collapse.training.losses import CustomInfoNCELoss
import torch

loss = CustomInfoNCELoss()
test_tensor = torch.Tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
])/10
labels = torch.Tensor([0, 1, 1, 2])

# Ground truth
print(loss(test_tensor, labels))

def info_nce(x):
    pass

# Test
print(info_nce(test_tensor))