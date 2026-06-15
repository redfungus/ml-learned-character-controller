using UnityEngine;

/// <summary>
/// Orientation controller using a trained neural network model (ONNX via Unity Sentis).
/// </summary>
public class LearnedOrientationController : OrientationController
{
    [Header("Model")]
    [Tooltip("Model settings asset containing the ONNX model")]
    public OrientationModelSettings settings;

    protected override RotationSpace rotationSpace => settings.rotationSpace;

    [Header("Inference Settings")]
    [Tooltip("Backend type for inference")]
    public Unity.InferenceEngine.BackendType backendType = Unity.InferenceEngine.BackendType.GPUCompute;

    [Tooltip("Always run model inference, even when input is zero")]
    public bool alwaysUseModel = false;

    [Header("Debug")]
    [Tooltip("Show debug information in console")]
    public bool debugMode = false;

    // Input tensor: [pad_x, pad_y, vel_x, vel_y, vel_z, dt]
    private const int InputFeatureCount = 6;

    private Unity.InferenceEngine.Model m_RuntimeModel;
    private Unity.InferenceEngine.Worker m_Worker;

    private Unity.InferenceEngine.Tensor<float> m_InputTensor;
    private float[] m_InputData = new float[InputFeatureCount];

    private bool m_PredictAngularVelocity;
    private bool m_OutputShapeDetected;

    protected override void Start()
    {
        base.Start();
        LoadModel();
    }

    private void OnDestroy()
    {
        m_Worker?.Dispose();
        m_InputTensor?.Dispose();
    }

    /// <summary>Load or reload the model.</summary>
    private void LoadModel()
    {
        if (settings == null || settings.modelAsset == null)
        {
            Debug.LogError("LearnedOrientationController: No model settings or model asset assigned!");
            enabled = false;
            return;
        }

        m_RuntimeModel = Unity.InferenceEngine.ModelLoader.Load(settings.modelAsset);
        m_Worker = new Unity.InferenceEngine.Worker(m_RuntimeModel, backendType);
        currentOrientation = QuaternionUtils.EnsurePositiveW(GetRotation());
        m_InputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, InputFeatureCount));
        m_OutputShapeDetected = false;

        Debug.Log("LearnedOrientationController: Model loaded successfully");
    }

    public override void UpdateOrientation(Vector2 input, float dt)
    {
        if (!ShouldUpdateOrientation(input))
        {
            angularVelocity = Vector3.zero;
            SetRotation(currentOrientation);
            return;
        }

        UpdateWithPrediction(input, dt);
        SetRotation(currentOrientation);
    }

    /// <summary>Check if orientation update is needed based on input and velocity.</summary>
    private bool ShouldUpdateOrientation(Vector2 input)
    {
        return alwaysUseModel || input.sqrMagnitude > 0f || IsAngularVelocitySignificant();
    }

    /// <summary>Run model prediction and update orientation state.</summary>
    private void UpdateWithPrediction(Vector2 input, float dt)
    {
        (Quaternion deltaQ, Vector3 predictedAngularVelocity) = PredictOrientationChange(input, dt);

        currentOrientation = rotationSpace == RotationSpace.Local
            ? currentOrientation * deltaQ
            : deltaQ * currentOrientation;

        angularVelocity = m_PredictAngularVelocity
            ? predictedAngularVelocity
            : QuaternionUtils.DeltaToAngularVelocity(deltaQ, dt);

        if (debugMode)
        {
            Debug.Log($"Model: dt={dt:F4}, AngVel={angularVelocity}, Orientation={currentOrientation}");
        }
    }

    /// <summary>Run neural network to predict orientation change.</summary>
    private (Quaternion, Vector3) PredictOrientationChange(Vector2 controlInput, float dt)
    {
        // Input: [pad_x, pad_y, vel_x, vel_y, vel_z, dt] (local frame - no orientation needed)
        m_InputData[0] = controlInput.x;
        m_InputData[1] = controlInput.y;
        m_InputData[2] = angularVelocity.x;
        m_InputData[3] = angularVelocity.y;
        m_InputData[4] = angularVelocity.z;
        m_InputData[5] = dt;

        m_InputTensor.Upload(m_InputData);
        m_Worker.Schedule(m_InputTensor);

        Unity.InferenceEngine.Tensor<float> outputTensor = m_Worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        outputTensor.ReadbackRequest();
        outputTensor.ReadbackAndClone();

        // Auto-detect output shape on first inference
        if (!m_OutputShapeDetected)
        {
            m_PredictAngularVelocity = outputTensor.shape[1] > 4;
            m_OutputShapeDetected = true;
            Debug.Log($"LearnedOrientationController: Auto-detected predictAngularVelocity={m_PredictAngularVelocity}");
        }

        // Output: [w, x, y, z] or [w, x, y, z, vel_x, vel_y, vel_z]
        Quaternion orientationChange = new Quaternion(
            outputTensor[0, 1],  // x
            outputTensor[0, 2],  // y
            outputTensor[0, 3],  // z
            outputTensor[0, 0]   // w
        );

        Vector3 predAngVel = Vector3.zero;
        if (m_PredictAngularVelocity)
        {
            predAngVel = new Vector3(
                outputTensor[0, 4],
                outputTensor[0, 5],
                outputTensor[0, 6]
            );
        }

        return (orientationChange.normalized, predAngVel);
    }

}
