import copy
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np
import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mydataset import CustomDataset
import matplotlib.pyplot as plt

torch.manual_seed(0) # https://pytorch.org/docs/stable/notes/randomness.html

    
X, y = make_classification(
            n_samples=10000, 
            n_features=2, 
            n_informative=2, 
            n_redundant=0, 
            n_clusters_per_class=2, 
            random_state=42,
            class_sep=2 # 4
        )

autoencoder_features_nb = 2
encoder = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, autoencoder_features_nb))
linear_classifier = nn.Linear(autoencoder_features_nb, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)



train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_test, y_test)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def display_model(model, data, labels):
    embeddings = model(torch.FloatTensor(data)).detach().numpy()
    print(embeddings.shape)
    plt.figure(figsize=(10, 10))
    plt.subplot(1,2,1)
    # plt.scatter(embeddings[:,0], [1]*len(embeddings), c=y_test_sample, alpha=0.2)
    if autoencoder_features_nb == 2:
        plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, alpha=0.2)
    else:
        plt.scatter(embeddings[:,0], [1]*len(embeddings), c=labels, alpha=0.2)
    plt.subplot(1,2,2)
    plt.scatter(data[:,0], data[:,1], c=labels, alpha=0.2)
    plt.show()

samples = np.arange(X_test.shape[0])
X_test_sample = X_test[samples]
y_test_sample = y_test[samples]
display_model(encoder, X_test_sample, y_test_sample)

class CustomCELoss(nn.Module):
    def __init__(self):
        super(CustomCELoss, self).__init__()

    def forward(self, predictions, targets):
        assert len(predictions) == len(targets), "Predictions and targets must have the same length."
        log_softmax = nn.LogSoftmax(dim=1)
        log_softmax_predictions = log_softmax(predictions)
        return -torch.sum(log_softmax_predictions[torch.arange(len(predictions)), targets])/len(predictions)
    

# source git: ...
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.02, contrast_mode='all',
                 base_temperature=0.02):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CustomSupConLoss(nn.Module):
    def __init__(self):
        super(CustomSupConLoss, self).__init__()
        self.temperature = 0.1

    def forward(self, embeddings, labels):
        assert len(embeddings) == len(labels), "Predictions and targets must have the same length."

        loss = 0

        for i in range(len(embeddings)):
            zi = embeddings[i]
            # use mask instead to optimize
            ais = torch.cat([embeddings[:i],embeddings[i+1:]])
            ais_targets = torch.cat([labels[:i], labels[i+1:]])
            pis = ais[ais_targets == labels[i]]

            

            # loss_i 
            # for pi in pis:
            #     loss += pi*zi/self.temperature
            loss += torch.sum(pis @ zi / self.temperature)/len(pis)

            loss -= torch.logsumexp(ais @ zi / self.temperature, dim=0) # dim = 0
                
            # print(predictions[i], targets[i])
            # print()

        return -loss/len(embeddings)*self.temperature
        # assert len(predictions) == len(targets), "Predictions and targets must have the same length."
        # log_softmax = nn.LogSoftmax(dim=1)
        # log_softmax_predictions = log_softmax(predictions)
        # return -torch.sum(log_softmax_predictions[torch.arange(len(predictions)), targets])/len(predictions)

class AutoencoderClassifier(L.LightningModule):
    def __init__(self, encoder, linear_classifier):
        super().__init__()
        self.encoder = encoder
        # self.linear_classifier = linear_classifier

    def forward(self, x):
        x = self.encoder(x)
        # x = self.linear_classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.shape)
        # print(y)
        # loss = nn.functional.cross_entropy(y_hat, y)
        # print()
        # print("Pytorch", loss)
        loss = CustomSupConLoss()(y_hat, y)
        # print(y_hat.unsqueeze(1).shape)
        # loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
        # print("Custom", loss2)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer



classifier = AutoencoderClassifier(encoder, linear_classifier)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model=classifier, train_dataloaders=train_dataloader)

# exit()
# checkpoint = "./lightning_logs/version_72/checkpoints/epoch=99-step=10500.ckpt"
from natsort import natsorted
checkpoint_folder = natsorted(list(os.listdir("./lightning_logs/")))[-1]
print(checkpoint_folder)
checkpoint = "./lightning_logs/{}/checkpoints/epoch=99-step=10500.ckpt".format(checkpoint_folder)

# checkpoint = "./lightning_logs/version_62/checkpoints/epoch=99-step=10500.ckpt"
# 
# checkpoint = "./lightning_logs/version_54/checkpoints/epoch=99-step=10500.ckpt"
linear_classifier = AutoencoderClassifier.load_from_checkpoint(checkpoint, encoder=encoder, linear_classifier=linear_classifier)
# linear_classifier = classifier
encoder = linear_classifier.encoder
encoder.eval()
linear_classifier.eval()

# samples = np.random.choice(X_test.shape[0], size=int(0.1*X_test.shape[0]), replace=False)

# plot the embeddings
# display_model(classifier.encoder, X_test_sample)
display_model(linear_classifier.encoder, X_test_sample, y_test_sample)

checkpoint = "./lightning_logs/version_66/checkpoints/epoch=72-step=7665.ckpt"
linear_classifier = AutoencoderClassifier.load_from_checkpoint(checkpoint, encoder=encoder, linear_classifier=linear_classifier)
# linear_classifier = classifier
encoder = linear_classifier.encoder
encoder.eval()
linear_classifier.eval()
display_model(linear_classifier.encoder, X_test_sample, y_test_sample)

# print(autoencoder(em))

# # predict classes
# y_hat = linear_classifier(torch.FloatTensor(X_test))
# y_hat = torch.argmax(y_hat, dim=1)
# # compute score
# score = (y_hat == torch.LongTensor(y_test)).sum().item() / len(y_test)
# print(score)

# implement supcon loss
# https://github.com/KevinMusgrave/pytorch-metric-learning/issues/281
# https://github.com/ivanpanshin/SupCon-Framework
# pytorch lightning