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

def infonce2(x, labels, temperature=0.1):
    embeddings = x
    num = torch.matmul(embeddings, embeddings.T) / temperature

    numerator_spread = -1 * torch.diagonal(num, 0)

    # Use torch.gather to index and select relevant elements
    mask = labels.unsqueeze(1) == labels  # Create a boolean mask
    denominator_spread = torch.logsumexp(num[mask], dim=0)


    log_prob_spread = numerator_spread + denominator_spread

    return log_prob_spread.mean()
print(infonce2(test_tensor, labels))

def infonce(x, labels, temperature=0.1):
    num = torch.matmul(x, x.T) / temperature
    print(num.shape)
    numerator_spread = -1 * torch.diagonal(num, 0)
    print(numerator_spread.shape)
    print("numerator", num)
    print(torch.logsumexp(num[1][labels==labels[1]], 0))
    denominator_spread = torch.stack(
        [
            torch.logsumexp(num[i][labels==labels[i]], 0)
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