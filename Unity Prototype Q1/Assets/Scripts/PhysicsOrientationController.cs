using UnityEngine;

/// <summary>
/// Controls object orientation using Xbox controller left stick, replicating the physics model from generate_data.py.
/// </summary>
public class PhysicsOrientationController : OrientationController
{
    [Header("Settings")]
    [Tooltip("Settings asset containing rotation space configuration")]
    public OrientationModelSettings settings;

    protected override RotationSpace rotationSpace => settings.rotationSpace;

    [Header("Control Settings")]
    [Tooltip("Control sensitivity multiplier (angular acceleration in rad/s²)")]
    public float sensitivity = 10.0f;

    [Tooltip("Angular velocity damping factor (0-1). Higher values = less friction")]
    [Range(0f, 1f)]
    public float damping = 0.95f;

    /// <summary>Convert 2D control input to angular acceleration.</summary>
    private Vector3 ControlInputToAngularAcceleration(Vector2 input)
    {
        return new Vector3(
            -input.y * sensitivity,  // Pitch (around x-axis)
            0f,                       // Roll (around y-axis) - no input
            input.x * sensitivity     // Yaw (around z-axis)
        );
    }

    /// <summary>Integrate orientation using semi-implicit Euler with damping.</summary>
    public override void UpdateOrientation(Vector2 input, float dt)
    {
        Vector3 angularAccel = ControlInputToAngularAcceleration(input);

        angularVelocity += angularAccel * dt;
        angularVelocity *= damping;

        if (!IsAngularVelocitySignificant())
        {
            angularVelocity = Vector3.zero;
        }

        bool localFrame = rotationSpace == RotationSpace.Local;
        Quaternion qDot = QuaternionUtils.AngularVelocityToDerivative(currentOrientation, angularVelocity, localFrame);
        currentOrientation.x += qDot.x * dt;
        currentOrientation.y += qDot.y * dt;
        currentOrientation.z += qDot.z * dt;
        currentOrientation.w += qDot.w * dt;

        currentOrientation = currentOrientation.normalized;
        SetRotation(currentOrientation);
    }
}
