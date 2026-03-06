import os
import numpy as np
import pandas as pd
from scipy.signal import lfilter

class MasterGenerator:
    def __init__(self, n_samples=100, T=1000, window=75):
        self.n_samples, self.T, self.window = n_samples, T, window
        self.cp = T // 2

    def save(self, X, name):
        os.makedirs("data/synthetic", exist_ok=True)
        y = np.full(self.n_samples, self.cp)
        df = pd.DataFrame(np.hstack((y.reshape(-1, 1), X)),
                          columns=['changepoint_index'] + [f't_{i}' for i in range(self.T)])
        df.to_csv(f"data/synthetic/{name}.csv", index=False)
        print(f"make: {name}.csv")

    def run_all(self):

        X = np.random.normal(0, 1, (self.n_samples, self.T))
        X[:, self.cp:] += 10.0
        self.save(X, "mean_abrupt")

        X = np.zeros((self.n_samples, self.T))
        X[:, :self.cp] = np.random.normal(0, 1.0, (self.n_samples, self.cp))
        X[:, self.cp:] = np.random.normal(0, 5.0, (self.n_samples, self.T - self.cp))
        self.save(X, "var_abrupt")

        X = np.zeros((self.n_samples, self.T))
        for i in range(self.n_samples):
            X[i, :self.cp] = lfilter([1.0], [1.0, -0.95], np.random.normal(0, 0.3, self.cp))
            X[i, self.cp:] = lfilter([1.0], [1.0, 0.95], np.random.normal(0, 0.3, self.T - self.cp))
        self.save(X, "ar_abrupt")

        X = np.zeros((self.n_samples, self.T)); t = np.arange(self.T)
        for i in range(self.n_samples):
            X[i] = np.where(t < self.cp, np.sin(2*np.pi*0.01*t), np.sin(2*np.pi*0.2*t)) + np.random.normal(0, 0.1, self.T)
        self.save(X, "quasi_abrupt")

        X = np.zeros((self.n_samples, self.T))
        X[:, :self.cp] = np.random.normal(0, 1, (self.n_samples, self.cp))
        X[:, self.cp:] = np.random.standard_cauchy((self.n_samples, self.T - self.cp))
        self.save(X, "cauchy_abrupt")

        X = np.random.normal(0, 1, (self.n_samples, self.T))
        ramp = np.linspace(0, 5.0, self.window)
        for i in range(self.n_samples):
            X[i, self.cp : self.cp + self.window] += ramp
            X[i, self.cp + self.window :] += 5.0
        self.save(X, "mean_smooth")

        X = np.zeros((self.n_samples, self.T)); sigmas = np.linspace(1.0, 4.0, self.window)
        for i in range(self.n_samples):
            X[i, :self.cp] = np.random.normal(0, 1.0, self.cp)
            for t_idx, s in enumerate(sigmas): X[i, self.cp + t_idx] = np.random.normal(0, s)
            X[i, self.cp + self.window :] = np.random.normal(0, 4.0, self.T - (self.cp + self.window))
        self.save(X, "var_smooth")

        X = np.zeros((self.n_samples, self.T)); alphas = np.linspace(-0.95, 0.95, self.window)
        for i in range(self.n_samples):
            data = np.zeros(self.T); a = -0.95
            for t in range(1, self.T):
                if self.cp <= t < self.cp + self.window: a = alphas[t - self.cp]
                elif t >= self.cp + self.window: a = 0.95
                data[t] = a * data[t-1] + np.random.normal(0, 0.3)
            X[i] = data

        self.save(X, "ar_smooth")
        X = np.zeros((self.n_samples, self.T))
        for i in range(self.n_samples):
            freqs = np.zeros(self.T); freqs[:self.cp] = 0.01
            freqs[self.cp : self.cp+self.window] = np.linspace(0.01, 0.1, self.window)
            freqs[self.cp+self.window:] = 0.1
            X[i] = np.sin(np.cumsum(2 * np.pi * freqs)) + np.random.normal(0, 0.1, self.T)
        self.save(X, "quasi_smooth")

        X = np.random.normal(0, 1, (self.n_samples, self.T))
        for i in range(self.n_samples):
            mix = np.linspace(0, 1, self.window)
            for t_idx, m in enumerate(mix):
                X[i, self.cp + t_idx] = (1-m)*np.random.normal(0,1) + m*np.random.standard_cauchy()
            X[i, self.cp + self.window:] = np.random.standard_cauchy(self.T - self.cp - self.window)
        self.save(X, "cauchy_smooth")

if __name__ == "__main__":
    MasterGenerator().run_all()