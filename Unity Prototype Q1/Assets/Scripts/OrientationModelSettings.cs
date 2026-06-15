using UnityEngine;

/// <summary>
/// Defines whether rotations are applied in world space (global) or relative to the parent (local).
/// </summary>
public enum RotationSpace
{
    Global,  // World space rotation (transform.rotation)
    Local    // Parent-relative rotation (transform.localRotation)
}

/// <summary>
/// ScriptableObject that holds the ONNX model and settings for orientation prediction.
/// Create via Assets > Create > Orientation > Model Settings.
/// </summary>
[CreateAssetMenu(fileName = "OrientationModel", menuName = "Orientation/Model Settings")]
public class OrientationModelSettings : ScriptableObject
{
    [Tooltip("The trained ONNX model asset")]
    public Unity.InferenceEngine.ModelAsset modelAsset;

    [Tooltip("Apply rotations in world space (Global) or relative to parent (Local)")]
    public RotationSpace rotationSpace = RotationSpace.Global;
}
