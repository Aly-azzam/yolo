from __future__ import annotations


def angle_distance(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 180 - diff)


def run_dtw(
    expert_angles: list[float],
    learner_angles: list[float],
    window_ratio: float = 0.1,
) -> dict:
    n = len(expert_angles)
    m = len(learner_angles)

    window = max(1, int(max(n, m) * window_ratio))

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            cost = angle_distance(expert_angles[i - 1], learner_angles[j - 1])
            dp[i][j] = cost + min(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1],
            )

    # Backtrack
    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        directions = [
            (dp[i - 1][j], i - 1, j),
            (dp[i][j - 1], i, j - 1),
            (dp[i - 1][j - 1], i - 1, j - 1),
        ]

        _, i, j = min(directions, key=lambda x: x[0])

    path.reverse()

    matches = []
    for i, j in path:
        diff = angle_distance(expert_angles[i], learner_angles[j])
        matches.append(
            {
                "expert_index": i,
                "learner_index": j,
                "expert_angle": expert_angles[i],
                "learner_angle": learner_angles[j],
                "angle_difference": diff,
            }
        )

    total_cost = dp[n][m]
    normalized = total_cost / max(len(path), 1)

    return {
        "dtw_distance": total_cost,
        "normalized_distance": normalized,
        "path": path,
        "matches": matches,
    }
