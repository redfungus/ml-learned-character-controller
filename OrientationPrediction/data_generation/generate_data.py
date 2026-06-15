import argparse
import csv
from typing import Tuple

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (q1 * q2). Format: [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def angular_velocity_to_quaternion_derivative(
    q: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    """
    Convert angular velocity to quaternion derivative.
    q: quaternion [w, x, y, z]
    omega: angular velocity vector [wx, wy, wz] in radians/second
    """
    # Create pure quaternion from angular velocity
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

    # q_dot = 0.5 * omega_quat * q
    q_dot = 0.5 * quaternion_multiply(omega_quat, q)

    return q_dot


def control_input_to_angular_acceleration(
    control_input: np.ndarray, sensitivity: float = 10.0
) -> np.ndarray:
    """
    Convert 2D control pad input to angular acceleration.
    control_input: [x, y] values typically in range [-1, 1]
    Returns: angular acceleration [ax, ay, az] in radians/second^2
    """
    # Map 2D control to pitch and yaw angular acceleration
    # x-axis controls yaw (rotation around z-axis)
    # y-axis controls pitch (rotation around x-axis)
    angular_accel = np.array(
        [
            -control_input[1] * sensitivity,  # Pitch (around x-axis)
            0.0,  # Roll (around y-axis) - no input
            control_input[0] * sensitivity,  # Yaw (around z-axis)
        ]
    )

    return angular_accel


def integrate_orientation(
    current_orientation: np.ndarray,
    angular_velocity: np.ndarray,
    control_input: np.ndarray,
    dt: float = 1.0 / 60.0,
    sensitivity: float = 10.0,
    damping: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform forward integration of orientation quaternion.

    Parameters:
    -----------
    current_orientation : np.ndarray
        Current orientation as quaternion [w, x, y, z]
    angular_velocity : np.ndarray
        Current angular velocity [wx, wy, wz] in radians/second
    control_input : np.ndarray
        2D control pad input [x, y], typically in range [-1, 1]
    dt : float
        Time step in seconds (default: 1/60 for 60 FPS)
    sensitivity : float
        Control sensitivity multiplier
    damping : float
        Angular velocity damping factor (0-1)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (new_orientation, new_angular_velocity)
    """
    # 1. Convert control input to angular acceleration
    angular_accel = control_input_to_angular_acceleration(control_input, sensitivity)

    # 2. Update angular velocity using semi-implicit Euler
    new_angular_velocity = angular_velocity + angular_accel * dt
    new_angular_velocity *= damping  # Apply damping

    # 3. Compute quaternion derivative from angular velocity
    q_dot = angular_velocity_to_quaternion_derivative(
        current_orientation, new_angular_velocity
    )

    # 4. Integrate quaternion using Euler method
    new_orientation = current_orientation + q_dot * dt

    # 5. Normalize to prevent drift
    new_orientation = normalize_quaternion(new_orientation)

    return new_orientation, new_angular_velocity


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


def quaternion_difference(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion representing the rotation from q1 to q2.
    Returns q_diff such that q2 = q_diff * q1
    """
    # Conjugate of q1
    q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    # q_diff = q2 * q1_conjugate
    return quaternion_multiply(q2, q1_conj)


def generate_training_data(
    num_samples: int,
    output_file: str,
    dt_min: float = 0.01,
    dt_max: float = 0.05,
    sensitivity: float = 10.0,
    damping: float = 0.95,
    random_seed: int | None = None,
):
    """
    Generate training data for orientation prediction.

    Parameters:
    -----------
    num_samples : int
        Number of data samples to generate
    output_file : str
        Path to output CSV file
    dt_min : float
        Minimum timestep in seconds
    dt_max : float
        Maximum timestep in seconds
    sensitivity : float
        Control sensitivity multiplier
    damping : float
        Angular velocity damping factor
    random_seed : int
        Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Prepare CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = [
            "current_orientation_w",
            "current_orientation_x",
            "current_orientation_y",
            "current_orientation_z",
            "pad_input_x",
            "pad_input_y",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "timestep_seconds",
            "orientation_change_w",
            "orientation_change_x",
            "orientation_change_y",
            "orientation_change_z",
        ]
        writer.writerow(header)

        print(f"Generating {num_samples} training samples...")
        print(f"Timestep range: [{dt_min:.4f}, {dt_max:.4f}] seconds")
        print(f"Sensitivity: {sensitivity}, Damping: {damping}")
        print(f"Output file: {output_file}")
        print()

        # Initialize starting state
        current_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        current_angular_velocity = np.array([0.0, 0.0, 0.0])

        for i in range(num_samples):
            # Random timestep
            dt = np.random.uniform(dt_min, dt_max)

            # dt = 1/60.0  # fixed timestep for simplicity

            # Random control input (typically in range [-1, 1])
            control_input = np.random.uniform(-1.0, 1.0, size=2)

            # example contol_input with only one axis active deterministic
            # control_input = np.array([1.0, 0.0])

            # Store current state before integration
            prev_orientation = current_orientation.copy()
            prev_angular_velocity = current_angular_velocity.copy()

            # Perform integration
            current_orientation, current_angular_velocity = integrate_orientation(
                current_orientation,
                current_angular_velocity,
                control_input,
                dt,
                sensitivity,
                damping,
            )

            # Calculate orientation change (quaternion difference)
            orientation_change = quaternion_difference(
                prev_orientation, current_orientation
            )

            # Write row to CSV
            row = [
                # Current orientation (before update)
                prev_orientation[0],
                prev_orientation[1],
                prev_orientation[2],
                prev_orientation[3],
                # Control pad input
                control_input[0],
                control_input[1],
                # Angular velocity (before update)
                prev_angular_velocity[0],
                prev_angular_velocity[1],
                prev_angular_velocity[2],
                # Timestep
                dt,
                # Expected change in orientation
                orientation_change[0],
                orientation_change[1],
                orientation_change[2],
                orientation_change[3],
            ]
            writer.writerow(row)

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")

    print(f"\nSuccessfully generated {num_samples} samples to {output_file}")


def main():
    """Main function to parse arguments and generate training data."""
    parser = argparse.ArgumentParser(
        description="Generate training data for quaternion orientation prediction"
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10000,
        help="Number of training samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="orientation_training_data.csv",
        help="Output CSV file path (default: orientation_training_data.csv)",
    )
    parser.add_argument(
        "--dt-min",
        type=float,
        default=0.01,
        help="Minimum timestep in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--dt-max",
        type=float,
        default=0.05,
        help="Maximum timestep in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=10.0,
        help="Control sensitivity multiplier (default: 10.0)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.95,
        help="Angular velocity damping factor (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    # Generate training data
    generate_training_data(
        num_samples=args.num_samples,
        output_file=args.output,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        sensitivity=args.sensitivity,
        damping=args.damping,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
