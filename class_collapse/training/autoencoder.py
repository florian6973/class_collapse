from torch import nn, optim
import lightning as L
import torch
from class_collapse.config.config import Config
from class_collapse.training.losses import CustomInfoNCELoss, SupConLoss, CustomCELoss

from torch.utils.data import Dataset, DataLoader

class Autoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, config: Config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_values = []
        self.current_loss_values = []
        self.config = config

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x = x + torch.randn_like(x) * self.config.hydra_config["loss"]["augmentation"]
        y_hat = self(x)
        if "temperature" in self.config.hydra_config["loss"]:
            temperature = self.config.hydra_config["loss"]["temperature"]
        if self.config.hydra_config["loss"]["name"] == "MSE_loss":
            x_hat = self.decoder(y_hat)
            loss = nn.functional.mse_loss(x_hat, x)
        elif self.config.hydra_config["loss"]["name"] == "supcon_2020":
            loss = SupConLoss(temperature=temperature)(y_hat.unsqueeze(1), labels=y)
        elif self.config.hydra_config["loss"]["name"] == "nce":
            loss = CustomInfoNCELoss(temperature=temperature)(y_hat, y) # diverges
        elif self.config.hydra_config["loss"]["name"] == "spread":
            alpha = self.config.hydra_config["loss"]["alpha"]
            loss = (1-alpha)*SupConLoss(temperature=temperature)(y_hat.unsqueeze(1), labels=y) + \
                    alpha*CustomInfoNCELoss(temperature=temperature)(y_hat, y)
        else:
            raise ValueError("Unknown loss")
        
        self.log("train_loss", loss)
        self.current_loss_values.append(loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.hydra_config["model"]["lr"])
        return optimizer
    
    def on_train_epoch_end(self):
        for loss in self.current_loss_values:
            self.loss_values.append(loss.item())
        self.current_loss_values.clear()  # free memory'

def get_model(config: Config, dataloader: DataLoader, dim_data) -> L.LightningModule:
    autoencoder_features_nb = config.hydra_config["model"]["embeddings_features"]
    autoencoder_features_hidden = config.hydra_config["model"]["embeddings_hidden"]

    encoder = nn.Sequential(nn.Linear(dim_data, autoencoder_features_hidden), 
                            nn.ReLU(), 
                            nn.Linear(autoencoder_features_hidden, autoencoder_features_nb))
    
    decoder = nn.Sequential(nn.Linear(autoencoder_features_nb, autoencoder_features_hidden), 
                            nn.ReLU(), 
                            nn.Linear(autoencoder_features_hidden, dim_data))

    if config.hydra_config["model"]["name"] == "encoder_only":
        model = Autoencoder(encoder, decoder, config)
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

