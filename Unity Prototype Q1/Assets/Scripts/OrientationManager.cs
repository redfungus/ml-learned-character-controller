using UnityEngine;

/// <summary>
/// Centralized manager that reads input once in Update() and drives both orientation controllers
/// identically in FixedUpdate(), ensuring synchronized behavior for comparison.
/// </summary>
public class OrientationManager : MonoBehaviour
{
    [Header("Controllers")]
    [Tooltip("Physics-based orientation controller")]
    public OrientationController physicsController;

    [Tooltip("Learned (neural network) orientation controller")]
    public OrientationController learnedController;

    [Header("Input Settings")]
    [Tooltip("Input deadzone - values below this are treated as zero")]
    public float deadzone = 0.1f;

    private Vector2 cachedInput;
    private bool cachedResetPressed;

    private OrientationController[] controllers;

    private void Awake()
    {
        // Build array of non-null controllers for iteration
        int count = (physicsController != null ? 1 : 0) + (learnedController != null ? 1 : 0);
        controllers = new OrientationController[count];
        int i = 0;
        if (physicsController != null) controllers[i++] = physicsController;
        if (learnedController != null) controllers[i++] = learnedController;
    }

    private void Update()
    {
        cachedInput = GetControlInput();

        // Latch reset (B button on Xbox controller or R key)
        if (Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.JoystickButton1))
            cachedResetPressed = true;
    }

    /// <summary>Get control input from player or random generation.</summary>
    private Vector2 GetControlInput()
    {
        // Check for random input mode (Spacebar or Y button on Xbox controller)
        bool randomInputActive = Input.GetKey(KeyCode.Space) || Input.GetKey(KeyCode.JoystickButton3);

        if (randomInputActive)
        {
            return new Vector2(Random.Range(-1f, 1f), Random.Range(-1f, 1f));
        }

        return GetPlayerInput();
    }

    /// <summary>Read player input with deadzone applied.</summary>
    private Vector2 GetPlayerInput()
    {
        float x = Input.GetAxis("Horizontal");
        float y = Input.GetAxis("Vertical");

        if (Mathf.Abs(x) < deadzone) x = 0f;
        if (Mathf.Abs(y) < deadzone) y = 0f;

        return new Vector2(x, y);
    }

    private void FixedUpdate()
    {
        if (cachedResetPressed)
        {
            cachedResetPressed = false;
            foreach (var controller in controllers)
                controller.ResetToStart();
            return;
        }

        float dt = Time.fixedDeltaTime;
        foreach (var controller in controllers)
            controller.UpdateOrientation(cachedInput, dt);
    }
}
