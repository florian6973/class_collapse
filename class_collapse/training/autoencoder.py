from torch import nn, optim
import lightning as L
from class_collapse.config.config import Config
from class_collapse.training.losses import CustomInfoNCELoss, CustomSupConLoss, SupConLoss, CustomCELoss

from torch.utils.data import Dataset, DataLoader

class Autoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, config: Config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
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
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        if self.config.hydra_config["loss"]["name"] == "MSE_loss":
            x_hat = self.decoder(y_hat)
            loss = nn.functional.mse_loss(x_hat, x)
        # elif self.config.hydra_config["loss"]["name"] == "CE_loss":
        #     loss = nn.functional.cross_entropy(y_hat, y)
        elif self.config.hydra_config["loss"]["name"] == "supcon_2020":
            loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
        elif self.config.hydra_config["loss"]["name"] == "nce":
            loss = CustomInfoNCELoss()(y_hat, y) # diverges
        elif self.config.hydra_config["loss"]["name"] == "spread":
            alpha = self.config.hydra_config["loss"]["alpha"]
            loss = (1-alpha)*SupConLoss()(y_hat.unsqueeze(1), labels=y) + \
                    alpha*CustomInfoNCELoss()(y_hat, y)
        else:
            raise ValueError("Unknown loss")
        # print(y_hat.shape)
        # print(y)
        # loss = nn.functional.cross_entropy(y_hat, y)
        # print()
        # print("Pytorch", loss)
        # loss = CustomSupConLoss()(y_hat, y)
        # print(y_hat.unsqueeze(1).shape)
        # loss = SupConLoss()(y_hat.unsqueeze(1), labels=y)
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
    autoencoder_features_hidden = config.hydra_config["model"]["embeddings_hidden"]
    encoder = nn.Sequential(nn.Linear(2, autoencoder_features_hidden), 
                            nn.ReLU(), 
                            nn.Linear(autoencoder_features_hidden, autoencoder_features_nb))
    # linear_classifier = nn.Linear(autoencoder_features_nb, 2)
    decoder = nn.Sequential(nn.Linear(autoencoder_features_nb, autoencoder_features_hidden), 
                            nn.ReLU(), 
                            nn.Linear(autoencoder_features_hidden, 2))

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

