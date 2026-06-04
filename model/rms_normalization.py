import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_arr=np.array(x,dtype=float)
        gamma_arr=np.array(gamma,dtype=float)
        rms=np.sqrt(np.mean(x_arr**2)+eps)
        x_hat=x_arr/rms
        output=x_hat*gamma_arr
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        return np.round(output, 4).tolist()
        pass
