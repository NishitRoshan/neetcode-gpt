import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)

        W2 = np.array(W2)
        b2 = np.array(b2)

        y_true = np.array(y_true)

        # Forward pass
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)

        y_pred = np.dot(W2, a1) + b2

        loss = np.mean((y_pred - y_true) ** 2)

        # Backward pass
        n = y_true.size

        dL_dy = (2 / n) * (y_pred - y_true)

        dW2 = np.outer(dL_dy, a1)
        db2 = dL_dy

        da1 = np.dot(W2.T, dL_dy)

        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),

            "dW1": (np.round(dW1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),

            "dW2": (np.round(dW2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist()
        }