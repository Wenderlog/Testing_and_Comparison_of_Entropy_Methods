import pandas as pd
import numpy as np
import os


def prepare_coal_mining_data():
    disasters = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2,
        1, 3, 2,
        2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1,
        1, 0, 2,
        3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
    ]

    changepoint_index = 39
    row = [changepoint_index] + disasters
    dataset = np.array([row] * 10)

    columns = ['changepoint_index'] + [f't_{i}' for i in range(len(disasters))]
    df = pd.DataFrame(dataset, columns=columns)

    os.makedirs("data/real", exist_ok=True)
    filepath = "data/real/coal_mining.csv"
    df.to_csv(filepath, index=False)
    print(f"dataset Coal Mining Disasters save in {filepath}")


if __name__ == "__main__":
    prepare_coal_mining_data()