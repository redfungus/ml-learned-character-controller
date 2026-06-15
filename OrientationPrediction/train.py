"""Main training script for orientation prediction."""

import argparse
import time

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import OrientationDataModule
from src.lightning_module import OrientationPredictionModule


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_kwargs(config: dict, input_dim: int, output_dim: int) -> dict:
    """Build model constructor kwargs from config."""
    model_config = config["model"]
    model_name = model_config.get("name", "mlp")

    # Common kwargs
    kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "dropout": model_config.get("dropout", 0.1),
        "normalize_output": model_config.get("normalize_output", True),
        "predict_angular_velocity": model_config.get("predict_angular_velocity", False),
    }

    if model_name == "split_encoder":
        # Split encoder specific params
        kwargs.update(
            {
                "state_encoder_dim": model_config.get("state_encoder_dim", 64),
                "control_encoder_dim": model_config.get("control_encoder_dim", 32),
                "decoder_dim": model_config.get("decoder_dim", 64),
                "embedding_dim": model_config.get("embedding_dim", 32),
                "num_encoder_layers": model_config.get("num_encoder_layers", 2),
                "num_decoder_layers": model_config.get("num_decoder_layers", 2),
            }
        )
    else:
        # MLP params
        kwargs.update(
            {
                "hidden_dim": model_config.get("hidden_dim", 128),
                "num_layers": model_config.get("num_layers", 2),
            }
        )

    return kwargs


def main():
    parser = argparse.ArgumentParser(description="Train orientation prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["training"].get("seed", 42)
    pl.seed_everything(seed)

    rollout_config = config["training"].get("rollout", {})
    rollout_enabled = rollout_config.get("enabled", False)
    rollout_steps = rollout_config.get("steps", 1) if rollout_enabled else 1
    predict_angular_velocity = config["model"].get("predict_angular_velocity", False)

    data_module = OrientationDataModule(
        data_path=config["data"]["train_path"],
        val_split=config["data"].get("val_split", 0.15),
        test_split=config["data"].get("test_split", 0.15),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        rollout_steps=rollout_steps,
        seed=seed,
        predict_angular_velocity=predict_angular_velocity,
    )
    data_module.prepare_data()
    data_module.setup()

    input_dim = data_module.input_dim
    output_dim = data_module.output_dim

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    if rollout_enabled:
        print(f"Rollout steps: {rollout_steps}")
    if predict_angular_velocity:
        print("Angular velocity prediction: enabled")

    model_kwargs = build_model_kwargs(config, input_dim, output_dim)
    module = OrientationPredictionModule(
        model_name=config["model"]["name"],
        model_kwargs=model_kwargs,
        loss_name=config["loss"]["name"],
        loss_kwargs={},
        learning_rate=config["training"]["learning_rate"],
        rollout_config=rollout_config,
        norm_loss_weight=config["loss"].get("norm_loss_weight", 0.01),
        predict_angular_velocity=predict_angular_velocity,
        angular_velocity_loss_weight=config["loss"].get("angular_velocity_weight", 1.0),
    )

    print(f"\nModel: {config['model']['name']}")
    print(f"Loss: {config['loss']['name']}")
    print(f"Parameters: {module.model.get_num_parameters():,}")

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=config["logging"].get("save_top_k", 3),
            filename="{epoch}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

# Create logger with experiment name that appends time
    experiment_name = f"{config['model']['name']}_{config['loss']['name']}"
    logger = TensorBoardLogger(
        save_dir="logs",
        name=config["logging"].get("project_name", "quaternion_prediction") + f"_{time.strftime('%y-%m-%d_%H-%M-%S')}", 
        version=experiment_name,
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["hardware"].get("accelerator", "auto"),
        devices=config["hardware"].get("devices", 1),
        precision=config["hardware"].get("precision", 32),
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"].get("log_every_n_steps", 10),
        deterministic=True,
    )

    checkpoint_path = config["training"].get("checkpoint", "")
    print("\nStarting training...")
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.fit(module, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(module, data_module)

    print("\nRunning test evaluation...")
    trainer.test(module, data_module)

    print(f"\nTraining complete. Logs saved to: {logger.log_dir}")
    print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
