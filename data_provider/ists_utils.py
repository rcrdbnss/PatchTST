import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DatasetBaselineISTS(Dataset):

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='M', train_only=False):
        self.freq = freq
        self.flag = 'valid' if flag == 'val' else flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        with open(self.root_path + "/" + self.data_path, "rb") as f:
            D = pickle.load(f)

        x_, spt_, exg_, y_, time_ = f'x_{self.flag}', f'spt_{self.flag}', f'exg_{self.flag}', f'y_{self.flag}', f'time_{self.flag}'
        self.X, self.Xt = adapter(D[x_], D[spt_], D[exg_], D['x_feat_mask'], True, False)
        # Move target feature from the first feature to the last feature
        self.X = np.concatenate([self.X[:, :, 1:], self.X[:, :, 0][:, :, np.newaxis]], axis=2)
        self.y = D[y_][:, :, np.newaxis]
        yt = pd.to_datetime(D[time_][:, 2])

        time_feats = self.freq.split(",")
        yt_ = []
        for i, feat in enumerate(time_feats):
            if feat == "M":
                self.Xt[:, :, i] = self.Xt[:, :, i] / 11 - 0.5
                yt_.append((yt.month.to_numpy() - 1)[:, np.newaxis] / 11 - 0.5)
            elif feat == "WY":
                self.Xt[:, :, i] = self.Xt[:, :, i] / 53 - 0.5
                yt_.append((yt.isocalendar().week.to_numpy(dtype=int) - 1)[:, np.newaxis] / 53 - 0.5)
        self.yt = np.concatenate(yt_, axis=1)[:, np.newaxis, :]
        assert np.min(self.yt) >= -0.5 and np.max(self.yt) <= 0.5
        return

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.Xt[index], self.yt[index]

    def __len__(self):
        return self.X.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def get_X_T(X, feat_mask):
    def f(X):
        feat_mask_ = np.array(feat_mask)
        X_arg = feat_mask_ == 0
        T_arg = feat_mask_ == 2
        X, T = X[:, :, X_arg], X[:, :, T_arg]
        return X, T

    if len(np.shape(X)) == 3:
        return f(X)
    elif len(np.shape(X)) == 4:
        X_list = X
        X, T = [], []
        for X_ in X_list:
            x, t = f(X_)
            X.append(x)
            T.append(t)
        X = np.concatenate(X, axis=2)
        T = np.concatenate(T, axis=2)
        return X, T
    else:
        return [], []


def adapter(X, X_spt, X_exg, feat_mask, E: bool, S: bool):
    X, T = get_X_T(X, feat_mask)
    X_spt, T_spt = get_X_T(X_spt, feat_mask)
    X_exg, T_exg = get_X_T(X_exg, feat_mask)
    X = [X]
    if S:
        X.append(X_spt)
    if E:
        X.append(X_exg)
    X = np.concatenate(X, axis=2)
    return X, T


if __name__ == "__main__":
    data = DatasetBaselineISTS(
        root_path="../ists/output/pickle",
        data_path="adbpo_all_dev_nan5_np48_nf6.pickle",
        freq="M"
    )

    x, y, xt, yt = data[0]
    print(x.shape, xt.shape, y.shape, yt.shape)
