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
print(loss(test_tensor, labels))

def infonce(x, labels, temperature=0.1):
    num = torch.matmul(x, x.T) / temperature
    print(num.shape)
    numerator_spread = -1 * torch.diagonal(num, 0)
    print(numerator_spread.shape)
    denominator_spread = torch.stack(
        [
            # reshape num with an extra dimension,
            # then take the sum over everything
            torch.logsumexp(num[i][labels==labels[i]], 0).sum()
            for i in range(len(x))
        ]
    )
    print(denominator_spread.shape)
    log_prob_spread = numerator_spread + denominator_spread
    print(log_prob_spread.shape)
    print(log_prob_spread)
    return log_prob_spread.mean()

print(infonce(test_tensor, labels))

# from info_nce import InfoNCE, info_nce

# loss = InfoNCE()
# output = loss(test_tensor, labels)