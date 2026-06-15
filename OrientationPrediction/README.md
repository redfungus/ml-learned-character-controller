# Quaternion Orientation Prediction for Character Controllers

An autoregressive model that predicts orientation changes (quaternion deltas) for a character controller based on current orientation, control pad input, angular velocity, and timestep. Built with PyTorch Lightning for training and exportable to ONNX for use in Unity.

## Project Structure

```
OrientationPrediction/
├── configs/
│   └── default.yaml           # Training configuration
├── data_generation/
│   ├── generate_data.py       # Generate raw training data
│   └── preprocess.py          # Quaternion continuity preprocessing
├── notebooks/
│   ├── data_analysis.ipynb    # Data visualization and exploration
│   └── evaluation.ipynb       # Model evaluation and analysis
├── src/
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset
│   │   └── datamodule.py      # Lightning DataModule
│   ├── models/
│   │   ├── base.py            # Base model class
│   │   ├── mlp.py             # MLP implementation
│   │   └── registry.py        # Model registry
│   ├── losses/
│   │   └── quaternion.py      # Loss function implementions
│   ├── training/
│   │   └── schedulers.py      # Teacher forcing schedulers
│   ├── utils/
│   │   └── quaternion.py      # Quaternion utilities
│   └── lightning_module.py    # Lightning training module
├── train.py                   # Main training script
├── export_onnx.py             # Export to ONNX for Unity
├── requirements.txt           # Dependencies
└── requirements-dev.txt       # Dev Dependencies
```

## Requirements

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Preparing the Data

### 1. Generate Raw Data

Generate training samples with simulated character controller physics:

```bash
mkdir data
python data_generation/generate_data.py --num-samples 50000 --output data/raw_data.csv --seed 42
```

Options:
- `--num-samples`, `-n`: Number of samples to generate (default: 10000)
- `--output`, `-o`: Output file path
- `--seed`: Random seed for reproducibility
- `--dt-min`, `--dt-max`: Timestep range in seconds (default: 0.01-0.05)
- `--sensitivity`: Control sensitivity (default: 10.0)
- `--damping`: Angular velocity damping (default: 0.95)

### 2. Preprocess Data

Ensure quaternion continuity across the dataset:

```bash
python data_generation/preprocess.py --input data/raw_data.csv --output data/orientation_training_data.csv
```

Options:
- `--input`, `-i`: Path to raw data CSV (required)
- `--output`, `-o`: Path to output preprocessed CSV (required)
- `--quiet`, `-q`: Suppress progress output

## Running Data Analysis

Explore and visualize the training data using the Jupyter notebook:

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

This notebook provides visualizations of:
- Quaternion distributions
- Control input patterns
- Angular velocity statistics
- Orientation trajectories

## Training

Train the model using the default configuration:

```bash
python train.py --config configs/default.yaml
```

Training options can be modified in `configs/default.yaml`:
- Model architecture (`mlp` or `split_encoder`)
- Loss function (`mse`, `antipodal_mse`, or `geodesic`)
- Learning rate, batch size, epochs
- Multi-step rollout training with teacher forcing

Logs and checkpoints are saved to `logs/quaternion_prediction_<timestamp>/`.

## Running Evaluation

Evaluate the trained model using the evaluation notebook:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

This notebook provides:
- Test set metrics
- Prediction visualizations
- Error analysis
- Multi-step rollout evaluation

## Export to ONNX

Export the trained model for use in Unity:

```bash
python export_onnx.py --output model.onnx
```

This automatically finds the best checkpoint in the logs directory. Options:
- `--checkpoint`: Path to specific checkpoint file or directory
- `--output`: Output ONNX file path (default: `model.onnx`)
- `--opset`: ONNX opset version
- `--log-dir`: Directory to search for checkpoints (default: `logs/quaternion_prediction`)

Example with specific checkpoint:

```bash
python export_onnx.py --checkpoint logs/quaternion_prediction_20240101_120000/mlp_antipodal_mse/checkpoints/epoch=99-val_loss=0.0001.ckpt --output model.onnx
```

The exported ONNX model can be imported into Unity.
