// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Evaluates balance heuristic for multiple importance sampling (MIS).
/// @param n0 Number of samples taken from the first sampling strategy.
/// @param p0 PDF for the first sampling strategy.
/// @param n1 Number of samples taken from the second sampling strategy.
/// @param p1 PDF for the second sampling strategy.
/// @return Weight for the contribution from the first strategy (p0).
public float mis_balance_heuristic(float n0, float p0, float n1, float p1)
{
    float q0 = n0 * p0;
    float q1 = n1 * p1;
    return q0 / (q0 + q1);
}

/// Evaluates power heuristic for multiple importance sampling (MIS).
/// @param n0 Number of samples taken from the first sampling strategy.
/// @param p0 PDF for the first sampling strategy.
/// @param n1 Number of samples taken from the second sampling strategy.
/// @param p1 PDF for the second sampling strategy.
/// @param beta Power exponent.
/// @return Weight for the contribution from the first strategy (p0).
public float mis_power_heuristic(float n0, float p0, float n1, float p1, float beta)
{
    float q0 = pow(n0 * p0, beta);
    float q1 = pow(n1 * p1, beta);
    return q0 / (q0 + q1);
}

/// Evaluates power heuristic with beta=2 for multiple importance sampling (MIS).
/// @param n0 Number of samples taken from the first sampling strategy.
/// @param p0 PDF for the first sampling strategy.
/// @param n1 Number of samples taken from the second sampling strategy.
/// @param p1 PDF for the second sampling strategy.
/// @return Weight for the contribution from the first strategy (p0).
public float mis_power_two_heuristic(float n0, float p0, float n1, float p1)
{
    float q0 = (n0 * p0) * (n0 * p0);
    float q1 = (n1 * p1) * (n1 * p1);
    return q0 / (q0 + q1);
}

public enum class MISHeuristic {
    balance,
    power,
    power_two,
};

/// Evaluates heuristic for multiple importance sampling (MIS).
/// @param n0 Number of samples taken from the first sampling strategy.
/// @param p0 PDF for the first sampling strategy.
/// @param n1 Number of samples taken from the second sampling strategy.
/// @param p1 PDF for the second sampling strategy.
/// @param heuristic MIS heuristic.
/// @param beta Exponent for the power heuristic.
/// @return Weight for the contribution from the first strategy (p0).
public float
mis_heuristic(float n0, float p0, float n1, float p1, MISHeuristic heuristic = MISHeuristic::balance, float beta = 2.0)
{
    switch (heuristic) {
    case MISHeuristic::balance:
        return mis_balance_heuristic(n0, p0, n1, p1);
    case MISHeuristic::power:
        return mis_power_heuristic(n0, p0, n1, p1, beta);
    case MISHeuristic::power_two:
        return mis_power_two_heuristic(n0, p0, n1, p1);
    }
    return 0.0;
}
