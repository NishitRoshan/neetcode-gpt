import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        a=x
        for i in range(len(weights)-1):
            z=np.dot(a,weights[i])+biases[i]
            a=np.maximum(0,z)
        output=np.dot(a,weights[-1])+biases[-1]
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        return np.round(output,5)
        pass
