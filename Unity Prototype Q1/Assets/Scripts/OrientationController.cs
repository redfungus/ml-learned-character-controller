using UnityEngine;

/// <summary>
/// Base class for orientation controllers. Exposes UpdateOrientation and ResetToStart
/// to be called by OrientationManager.
/// </summary>
public abstract class OrientationController : MonoBehaviour
{
    [Tooltip("Angular velocity below this threshold snaps to zero")]
    public float angularVelocityThreshold = 0.01f;

    protected abstract RotationSpace rotationSpace { get; }

    protected Quaternion currentOrientation;
    protected Vector3 angularVelocity;
    protected Vector3 startingPosition;
    protected Quaternion startingRotation;

    protected virtual void Start()
    {
        startingPosition = transform.position;
        startingRotation = GetRotation();
        currentOrientation = GetRotation();
        angularVelocity = Vector3.zero;
    }

    /// <summary>
    /// Update orientation based on input. Called by OrientationManager.
    /// Implement in derived classes.
    /// </summary>
    /// <param name="input">2D control input</param>
    /// <param name="dt">Delta time for this physics step</param>
    public abstract void UpdateOrientation(Vector2 input, float dt);

    /// <summary>Reset to starting position and rotation.</summary>
    public virtual void ResetToStart()
    {
        transform.position = startingPosition;
        currentOrientation = startingRotation;
        angularVelocity = Vector3.zero;
        SetRotation(currentOrientation);
    }

    /// <summary>Check if angular velocity is above the threshold.</summary>
    protected bool IsAngularVelocitySignificant()
    {
        return angularVelocity.sqrMagnitude >= angularVelocityThreshold * angularVelocityThreshold;
    }

    /// <summary>Get the current rotation based on rotation space setting.</summary>
    protected Quaternion GetRotation()
    {
        return rotationSpace == RotationSpace.Global ? transform.rotation : transform.localRotation;
    }

    /// <summary>Set the rotation based on rotation space setting.</summary>
    protected void SetRotation(Quaternion rotation)
    {
        if (rotationSpace == RotationSpace.Global)
            transform.rotation = rotation;
        else
            transform.localRotation = rotation;
    }
}
