import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
from class_collapse.config.config import Config
from class_collapse.data.dataloaders import make_dataloader
from class_collapse.training.autoencoder import train_model, get_model
from class_collapse.eval.plot_embeddings import plot_embeddings
from class_collapse.eval.plot_classification import plot_classification
from torchviz import make_dot


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    torch.manual_seed(cfg["model"]["seed"]) # https://pytorch.org/docs/stable/notes/randomness.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(hydra_config=cfg, device=device)
    print("Using device:", config.device)

    data = make_dataloader(config)

    model = get_model(config, data.train_dataloader, data.X_train.shape[1])
    
    if config.hydra_config["model"]["visualize"]:
        y = model(torch.randn(1, data.X_train.shape[1]))
        # need to install graphviz to have this working (dot command)
        make_dot(y.mean(), params=dict(model.named_parameters())).render("model", format="png")
    
    plot_embeddings(config, model, data.test_dataloader, "before training")
    plot_classification(config, model, data, "before training")

    model = train_model(config, model, data.train_dataloader)

    plot_embeddings(config, model, data.test_dataloader, "after training")
    plot_classification(config, model, data, "after training")





if __name__ == "__main__":
    main()