from torch import nn, optim
import lightning as L
from class_collapse.config.config import Config
from class_collapse.training.losses import CustomSupConLoss, SupConLoss, CustomCELoss

from torch.utils.data import Dataset, DataLoader

class AutoencoderClassifier(L.LightningModule):
    def __init__(self, encoder, linear_classifier, config: Config):
        super().__init__()
        self.encoder = encoder
        self.loss_values = []
        self.current_loss_values = []
        self.config = config
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
        # loss = CustomSupConLoss()(y_hat, y)
        # print(y_hat.unsqueeze(1).shape)
        loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
        # print("Custom", loss2)
        self.log("train_loss", loss)
        self.current_loss_values.append(loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.hydra_config["model"]["lr"])
        # optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer
    
    def on_train_epoch_end(self):
        for loss in self.current_loss_values:
            self.loss_values.append(loss.item())
        self.current_loss_values.clear()  # free memory'

def get_model(config: Config, dataloader: DataLoader) -> L.LightningModule:
    autoencoder_features_nb = config.hydra_config["model"]["embeddings_features"]
    encoder = nn.Sequential(nn.Linear(2, 10), 
                            nn.ReLU(), 
                            nn.Linear(10, autoencoder_features_nb))
    linear_classifier = nn.Linear(autoencoder_features_nb, 2)

    if config.hydra_config["model"]["name"] == "encoder_only":
        model = AutoencoderClassifier(encoder, linear_classifier, config)
    else:
        raise ValueError("Unknown model")
    
    model.eval()
    
    return model


def train_model(config: Config, model, dataloader: DataLoader) -> L.LightningModule:
    model.train()

    if config.hydra_config["model"]["train"]:
        trainer = L.Trainer(max_epochs=config.hydra_config['model']['epochs'], accelerator="auto", devices="auto", strategy="auto")
        trainer.fit(model=model, train_dataloaders=dataloader)

    model.eval()

    return model

