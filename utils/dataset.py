import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import minmax_scale
import os
import numpy as np
import pywt

class UTSDataset(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None) -> None:
        super().__init__()

        assert len(seqs) > win_len + seq_len - 2

        self.seq_len = seq_len
        self.win_len = win_len
        self.labels = labels

        self.seqs = self.time_delay_embedding(seqs, win_len)

        if labels is not None:
            self.labels = self.time_delay_embedding_for_label(labels, win_len)


    def __len__(self):
        return len(self.seqs) - self.seq_len + 1
    

    def time_delay_embedding(self, seqs, win_len):
        return torch.tensor(np.array([seqs[i-win_len : i] for i in range(win_len, len(seqs)+1)]), dtype=torch.float32)
    
    def time_delay_embedding_for_label(self, labels, win_len):
        return torch.tensor(labels[win_len-1:], dtype=torch.int)
    

    def __getitem__(self, index):
        if self.labels is not None:
            return self.seqs[index : index + self.seq_len], self.labels[index + self.seq_len - 1]
        else:
            return self.seqs[index : index + self.seq_len]
        

class DenoisedUTSDataset(UTSDataset):
    def __init__(self, seqs, win_len, seq_len, labels=None) -> None:
        denoised_seqs = DenoisedUTSDataset.wavelet_denoising(seqs)
        super().__init__(denoised_seqs, win_len, seq_len, labels)

    @classmethod
    def wavelet_denoising(cls, data: np.ndarray):
        db4 = pywt.Wavelet('db4')
        coeffs = pywt.wavedec(data, db4)
        coeffs[len(coeffs)-1] *= 0
        coeffs[len(coeffs)-2] *= 0
        meta = pywt.waverec(coeffs, db4)

        if len(data) % 2 == 1:
            meta = meta[:-1]

        return meta
    

class UTSDatasetWithHistorySliding(UTSDataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        super().__init__(seqs, win_len, seq_len, labels)

        self.seq_stride = seq_stride

    def __getitem__(self, index):
        end_index = index + self.seq_len
        start_index = end_index - 1 - (self.seq_len - 1) * self.seq_stride

        if start_index < 0:
            start_index %= self.seq_stride

        seqs = self.seqs[start_index : end_index : self.seq_stride]

        padding_len = self.seq_len - len(seqs)
        if padding_len > 0:
            padding = torch.zeros((padding_len, self.win_len))
            seqs = torch.cat((padding, seqs), dim=0)



        if self.labels is not None:
            return seqs, self.labels[end_index - 1]
        else:
            return seqs


class KAD_DisformerTrainSet(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        self.seq_len = seq_len
        self.win_len = win_len
        self.seq_stride = seq_stride

        self.context_flow = UTSDataset(seqs, win_len, seq_len, labels)
        self.denoised_context = DenoisedUTSDataset(seqs, win_len, seq_len, labels)
        self.history_flow = UTSDatasetWithHistorySliding(seqs, win_len, seq_len, labels, seq_stride)

    def __len__(self):
        return len(self.context_flow)


    def __getitem__(self, index):
        return self.context_flow[index][0], self.history_flow[index][0], self.denoised_context[index][0]



class KAD_DisformerTestSet(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        self.seq_len = seq_len
        self.win_len = win_len
        self.seq_stride = seq_stride

        self.context_flow = UTSDataset(seqs, win_len, seq_len, labels)
        self.history_flow = UTSDatasetWithHistorySliding(seqs, win_len, seq_len, labels, seq_stride)

    def __len__(self):
        return len(self.context_flow)


    def __getitem__(self, index):
        return self.context_flow[index][0], self.history_flow[index][0]
    

def train_test_split(data, train_ratio=0.8):
    train_cnt = round(train_ratio * len(data))
    train_data = data[:train_cnt]
    test_data = data[train_cnt:]
    return train_data, test_data


def load_csvs(csv_path):
    if os.path.isdir(csv_path):
        uids = [i[:-4] for i in os.listdir(csv_path) if i.endswith(".csv")]
        datasets = {uid: pd.read_csv(os.path.join(csv_path, uid+".csv")) for uid in uids}
    else:
        datasets = pd.read_csv(csv_path)

    return datasets


if __name__ == '__main__':
    import pandas as pd

    raw_ts = pd.read_csv("../data/train.csv")[['value', 'label']].to_numpy()
    dd = DenoisedUTSDataset(raw_ts[:, 0], 20, 120, raw_ts[:, 1])
    d = UTSDataset(raw_ts[:, 0], 20, 120, raw_ts[:, 1])


    raw_ts = np.arange(1, 10)
    print(raw_ts)

    d = UTSDatasetWithHistorySliding(raw_ts, 1, 3, seq_stride=1)
    d2 = UTSDatasetWithHistorySliding(raw_ts, 1, 3, seq_stride=3)

    print(d2[0], d2[6], sep='\n')