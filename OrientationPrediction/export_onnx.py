"""Export trained PyTorch model to ONNX format for Unity."""

import argparse
from pathlib import Path

import torch
import torch.onnx

from src.lightning_module import OrientationPredictionModule


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 9,
    input_names: list = None,
    output_names: list = None,
):
    """Export Lightning checkpoint to ONNX format."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module = OrientationPredictionModule(**checkpoint["hyper_parameters"])
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()

    model = module.model
    model.cpu()
    model.eval()

    hparams = module.hparams
    model_kwargs = hparams.get("model_kwargs", {})
    # Input: [pad_x, pad_y, vel_x, vel_y, vel_z, dt] = 6D (no orientation)
    # Output: [qw, qx, qy, qz] = 4D, or [qw, qx, qy, qz, vel_x, vel_y, vel_z] = 7D
    input_dim = model_kwargs.get("input_dim", 6)
    output_dim = model_kwargs.get("output_dim", 4)
    predict_angular_velocity = model_kwargs.get("predict_angular_velocity", False)

    print(f"Model: {hparams.get('model_name', 'unknown')}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if predict_angular_velocity:
        print("Angular velocity prediction: enabled")

    dummy_input = torch.randn(1, input_dim)
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    print(f"\nExporting to ONNX (opset version {opset_version})...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to: {output_path}")

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("Note: Install 'onnx' package for verification")

    print("\nTesting inference equivalence...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)

    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        diff = abs(pytorch_output.numpy() - ort_output).max()
        print(f"Max PyTorch/ONNX difference: {diff:.2e}")
        print(
            "Inference equivalence: PASSED"
            if diff < 1e-5
            else "WARNING: Outputs differ!"
        )
    except ImportError:
        print("Note: Install 'onnxruntime' for inference testing")

    return output_path


def find_best_checkpoint(log_dir: str) -> str:
    """Find the best checkpoint in a log directory."""
    log_path = Path(log_dir)

    # Look for checkpoints directory
    checkpoint_dirs = list(log_path.glob("**/checkpoints"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints directory found in {log_dir}")

    # Find checkpoint files
    checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        checkpoints.extend(list(ckpt_dir.glob("*.ckpt")))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Prefer checkpoints with "best" or lowest loss in name
    for ckpt in checkpoints:
        if "best" in ckpt.name.lower():
            return str(ckpt)

    # Return most recent
    return str(checkpoints[0])


def main():
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch model to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to checkpoint file (.ckpt) or log directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output path for ONNX file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=9,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/quaternion_prediction",
        help="Log directory to search for checkpoints",
    )

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if Path(checkpoint_path).is_dir():
            checkpoint_path = find_best_checkpoint(checkpoint_path)
    else:
        # Auto-find in log directory
        print(f"Searching for checkpoints in: {args.log_dir}")
        checkpoint_path = find_best_checkpoint(args.log_dir)

    print(f"Using checkpoint: {checkpoint_path}")

    # Export
    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=args.output,
        opset_version=args.opset,
    )

    print(f"\n{'='*50}")
    print("Export complete!")
    print(f"Copy '{args.output}' to your Unity project's Assets folder")
    print("Then assign it to the LearnedOrientationController's Model Asset field")


if __name__ == "__main__":
    main()
