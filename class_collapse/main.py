import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
from class_collapse.config.config import Config
from class_collapse.data.synthetic_dataset import generate_dataset
from class_collapse.training.autoencoder import train_model, get_model
from class_collapse.eval.plot_embeddings import plot_embeddings
from class_collapse.eval.plot_classification import plot_classification

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    torch.manual_seed(cfg["model"]["seed"]) # https://pytorch.org/docs/stable/notes/randomness.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(hydra_config=cfg, device=device)
    print("Using device:", config.device)

    X_train, X_test, y_train, y_test, train_dataloader, test_dataloader = generate_dataset(config)

    model = get_model(config, train_dataloader)
    
    plot_embeddings(config, model, test_dataloader, "before training")
    plot_classification(config, model, X_train, y_train, X_test, y_test, "before training")
    # exit()

    model = train_model(config, model, train_dataloader)

    plot_embeddings(config, model, test_dataloader, "after training")
    plot_classification(config, model, X_train, y_train, X_test, y_test, "after training")





if __name__ == "__main__":
    main()