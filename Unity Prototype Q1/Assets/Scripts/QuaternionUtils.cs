using UnityEngine;

/// <summary>
/// Static utility class for quaternion math operations used by orientation controllers.
/// </summary>
public static class QuaternionUtils
{
    /// <summary>Scale factor for quaternion derivative: q_dot = 0.5 * omega * q</summary>
    private const float QuaternionDerivativeScale = 0.5f;

    /// <summary>
    /// Convert angular velocity to quaternion derivative.
    /// Global (world-frame): q_dot = 0.5 * omega_quat * q
    /// Local (body-frame): q_dot = 0.5 * q * omega_quat
    /// </summary>
    public static Quaternion AngularVelocityToDerivative(Quaternion q, Vector3 omega, bool localFrame)
    {
        Quaternion omegaQuat = new Quaternion(omega.x, omega.y, omega.z, 0f);
        Quaternion qDot = localFrame ? q * omegaQuat : omegaQuat * q;
        return new Quaternion(
            qDot.x * QuaternionDerivativeScale,
            qDot.y * QuaternionDerivativeScale,
            qDot.z * QuaternionDerivativeScale,
            qDot.w * QuaternionDerivativeScale
        );
    }

    /// <summary>Convert delta quaternion to angular velocity (inverse of Euler integration).</summary>
    public static Vector3 DeltaToAngularVelocity(Quaternion deltaQ, float dt)
    {
        deltaQ = EnsurePositiveW(deltaQ);
        float scale = 2.0f / dt;
        return new Vector3(scale * deltaQ.x, scale * deltaQ.y, scale * deltaQ.z);
    }

    /// <summary>Negate all components of a quaternion.</summary>
    private static Quaternion Negate(Quaternion q)
    {
        return new Quaternion(-q.x, -q.y, -q.z, -q.w);
    }

    /// <summary>Flip quaternion to positive w hemisphere.</summary>
    public static Quaternion EnsurePositiveW(Quaternion q)
    {
        return q.w < 0f ? Negate(q) : q;
    }

    /// <summary>Ensure quaternion is in the same hemisphere as reference (positive dot product).</summary>
    public static Quaternion EnsureSameHemisphere(Quaternion q, Quaternion reference)
    {
        return Quaternion.Dot(q, reference) < 0f ? Negate(q) : q;
    }
}
