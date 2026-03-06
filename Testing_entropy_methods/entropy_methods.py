import numpy as np
from collections import deque
from scipy.stats import norm


class StandaloneWindowCPD:

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.is_calibrated = False
        self.fixed_mu = 0.0
        self.fixed_sigma = 1.0
        self.fixed_bins = np.linspace(-15, 15, 60)

    def update(self, observation) -> float:
        if isinstance(observation, np.ndarray):
            val = float(observation.item() if observation.size == 1 else observation[0])
        else:
            val = float(observation)

        self.window.append(val)

        if len(self.window) < self.window_size:
            return 0.0

        if not self.is_calibrated:
            data = np.array(self.window)
            self.fixed_mu = np.mean(data)
            self.fixed_sigma = np.std(data) or 1.0
            self.is_calibrated = True

        return self._calculate()

    def _calculate(self) -> float:
        raise NotImplementedError

    def _get_probabilities(self, data: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(data, bins=self.fixed_bins)
        probs = counts / len(data)
        return probs[probs > 0]

class ShannonEntropyCPD(StandaloneWindowCPD):
    def _calculate(self) -> float:
        probs = self._get_probabilities(np.array(self.window))
        return -np.sum(probs * np.log2(probs))


class RenyiEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, q: float = 2.0):
        super().__init__(window_size)
        self.q = q

    def _calculate(self) -> float:
        probs = self._get_probabilities(np.array(self.window))
        if abs(self.q - 1.0) < 1e-5: return -np.sum(probs * np.log2(probs))
        return (1 / (1 - self.q)) * np.log2(np.sum(probs ** self.q))


class TsallisEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, q: float = 2.0):
        super().__init__(window_size)
        self.q = q

    def _calculate(self) -> float:
        probs = self._get_probabilities(np.array(self.window))
        if abs(self.q - 1.0) < 1e-5: return -np.sum(probs * np.log2(probs))
        return (1 / (self.q - 1)) * (1 - np.sum(probs ** self.q))


class KLDEntropyCPD(StandaloneWindowCPD):
    def _calculate(self) -> float:
        data = np.array(self.window)
        half = len(data) // 2
        past, curr = data[:half], data[half:]

        p_past, _ = np.histogram(past, bins=self.fixed_bins)
        p_curr, _ = np.histogram(curr, bins=self.fixed_bins)

        p_past = (p_past + 1e-9) / (half + 1e-9 * len(self.fixed_bins))
        p_curr = (p_curr + 1e-9) / (half + 1e-9 * len(self.fixed_bins))
        return np.sum(p_curr * np.log2(p_curr / p_past))


class PermutationEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 3, delay: int = 1):
        super().__init__(window_size)
        self.m = m
        self.delay = delay

    def _calculate(self) -> float:
        data = np.array(self.window)
        n = len(data)
        patterns = np.array(
            [data[i: i + self.m * self.delay: self.delay] for i in range(n - (self.m - 1) * self.delay)])
        ordinals = np.argsort(patterns, axis=1)
        _, counts = np.unique(ordinals, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))


class BubbleEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 3):
        super().__init__(window_size)
        self.m = m

    def _calculate(self) -> float:
        data = np.array(self.window)
        n = len(data)
        patterns = np.array([data[i: i + self.m] for i in range(n - self.m + 1)])
        swaps = []
        for p in patterns:
            inv_count = sum(1 for i in range(self.m) for j in range(i + 1, self.m) if p[i] > p[j])
            swaps.append(inv_count)
        _, counts = np.unique(swaps, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))


class SlopeEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 3):
        super().__init__(window_size)
        self.m = m
        self.gamma = None
        self.delta = None

    def _calculate(self) -> float:
        if self.gamma is None:
            self.gamma = 1.5 * self.fixed_sigma
            self.delta = 0.5 * self.fixed_sigma

        data = np.array(self.window)
        diffs = np.diff(data)
        symbols = np.zeros_like(diffs, dtype=int)
        symbols[diffs > self.gamma] = 2
        symbols[(diffs <= self.gamma) & (diffs > self.delta)] = 1
        symbols[(diffs >= -self.delta) & (diffs <= self.delta)] = 0
        symbols[(diffs < -self.delta) & (diffs >= -self.gamma)] = -1
        symbols[diffs < -self.gamma] = -2

        n = len(symbols)
        patterns = np.array([symbols[i: i + self.m - 1] for i in range(n - self.m + 2)])
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))


class DispersionEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 2, c: int = 3):
        super().__init__(window_size)
        self.m = m
        self.c = c

    def _calculate(self) -> float:
        data = np.array(self.window)
        y = norm.cdf(data, loc=self.fixed_mu, scale=self.fixed_sigma)
        z = np.round(self.c * y + 0.5).astype(int)
        z = np.clip(z, 1, self.c)

        n = len(z)
        patterns = np.array([z[i: i + self.m] for i in range(n - self.m + 1)])
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

class ApproximateEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 2, r_ratio: float = 0.2):
        super().__init__(window_size)
        self.m = m
        self.r_ratio = r_ratio
        self.fixed_r = None

    def _phi(self, m: int, data: np.ndarray, r: float) -> float:
        n = len(data)
        patterns = np.array([data[i: i + m] for i in range(n - m + 1)])
        dist = np.max(np.abs(patterns[:, None] - patterns), axis=2)
        c = np.sum(dist <= r, axis=1) / (n - m + 1)
        return np.mean(np.log(c))

    def _calculate(self) -> float:
        data = np.array(self.window)
        if self.fixed_r is None:
            self.fixed_r = self.r_ratio * self.fixed_sigma

        phi_m = self._phi(self.m, data, self.fixed_r)
        phi_m_plus_1 = self._phi(self.m + 1, data, self.fixed_r)
        return phi_m - phi_m_plus_1


class SampleEntropyCPD(StandaloneWindowCPD):
    def __init__(self, window_size: int = 50, m: int = 2, r_ratio: float = 0.2):
        super().__init__(window_size)
        self.m = m
        self.r_ratio = r_ratio
        self.fixed_r = None

    def _calculate(self) -> float:
        data = np.array(self.window)
        if self.fixed_r is None:
            self.fixed_r = self.r_ratio * self.fixed_sigma

        def _get_B(m_val):
            n = len(data)
            patterns = np.array([data[i: i + m_val] for i in range(n - m_val + 1)])
            dist = np.max(np.abs(patterns[:, None] - patterns), axis=2)
            np.fill_diagonal(dist, np.inf)
            return np.sum(dist <= self.fixed_r)

        B = _get_B(self.m)
        A = _get_B(self.m + 1)
        if B == 0 or A == 0: return 0.0
        return -np.log(A / B)