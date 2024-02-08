from models.KAD_Disformer import KAD_Disformer
# %%
import torch
from torch import optim
from utils.dataset import KAD_DisformerTestSet, KAD_DisformerTrainSet
from torch.utils.data import DataLoader, ConcatDataset
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
from random import sample
import random

# %%
from utils.evaluate import best_f1_score_range, best_f1_score_point

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

setup_seed(2022)

# %%
device = 'cuda:0'


# %%
def pre_train(data_paths):
    d_model = 20
    seq_len = 100
    lr = 1e-3
    n_epoch = 20

    model = KAD_Disformer(N=0, d_model=d_model, layers=1)
    loss_function = KAD_Disformer.loss_function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def prepare_data(data_path):
        data_df = pd.read_csv(data_path)
        raw_series = data_df['value'].to_numpy()

        raw_series = minmax_scale(raw_series)

        train_dataset = KAD_DisformerTrainSet(raw_series, win_len=20, seq_len=100)
        dataloader = DataLoader(train_dataset, batch_size=256, drop_last=False, shuffle=True)

        return dataloader

    def train(dataloader, model):
        epoch_bar = tqdm(range(n_epoch))
        def train_one_epoch(step_bar):
            loss_sum = 0
            for step, (batch_X, batch_Y) in enumerate(step_bar):
                batch_X = batch_X.to(device).view(-1, seq_len, d_model)
                batch_Y = batch_Y.to(device).view(-1, seq_len, d_model)

                model.zero_grad()
                predicted = model(batch_X)
                loss = loss_function(predicted, batch_Y)
                loss_sum += loss.item()
                
                if step % 10 == 0:
                    step_bar.set_description(f"Loss: {loss_sum / 10}")
                    loss_sum = 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return model
        
        epoch_bar.set_description("Epoch")
        for epoch in epoch_bar:
            step_bar = tqdm(dataloader)
            model = model.to(device)

            model = train_one_epoch(step_bar)



    for data_path in tqdm(data_paths):
        print(f"Pre training dataset: {data_path}")

        dataloader = prepare_data(data_path)
        train(dataloader, model)

    
    return model
    

# %%
def get_params_to_update(model):
    meta_params = []
    for name, params in model.named_parameters():
        if "Wqm" in name or "Wkm" in name or "Wvm" in name:
            meta_params.append(params)

    return meta_params

def loopy(dl):
    while True:
        for x in iter(dl):
            yield x

# %%
def fine_tune(model, his_dataloader, dataloader):
    lr = 1e-3
    n_epoch = 20
    alpha = 0.2

    loss_function = KAD_Disformer.loss_function
    optimizer = optim.Adam(get_params_to_update(model), lr=lr)
    epoch_bar = tqdm(range(n_epoch))

    def train_one_epoch(step_bar):
        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(step_bar):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
        
            batch_X_his, batch_Y_his = loopy(his_dataloader)
            batch_X_his = batch_X_his.to(device)
            batch_Y_his = batch_Y_his.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            predicted = model(batch_X_his)
            loss1 = loss_function(predicted, batch_Y_his)
            
            loss1.backward(retain_graph=True)
            optimizer.step()

            predicted = model(batch_X)
            loss2 = loss_function(predicted, batch_Y)

            loss = alpha * loss1 + (1-alpha) * loss2
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()
            

            if step % 10 == 0:
                step_bar.set_description(f"loss {loss_sum / 10}")
                loss_sum = 0


            optimizer.step()

        return model
    
    epoch_bar.set_description("Fine-tune Epoch")
    for epoch in epoch_bar:
        step_bar = tqdm(dataloader)
        model = model.to(device)

        model = train_one_epoch(step_bar)

    
    return model
# %%
def test(model, data_paths):
    def prepare_data(data_path):
        data_df = pd.read_csv(data_path)
        labels = data_df['label'].to_numpy()
        raw_series = data_df['value'].to_numpy()

        raw_series = minmax_scale(raw_series)

        train_dataset = KAD_DisformerTrainSet(raw_series, win_len=20, seq_len=100)
        dataloader = DataLoader(train_dataset, batch_size=256, drop_last=False, shuffle=False)

        return dataloader, labels, raw_series

    def test_a_dataset(model, dataloader):
        reconstruct_ls = []
        device = 'cpu'

        with torch.no_grad():
            for step, (batch_x, _) in enumerate(tqdm(dataloader)):
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
                reconstruct_ls.append(reconstructed.to('cpu').numpy()[:, -1, -1])

        return np.concatenate(reconstruct_ls)


    res = []
    for data_path in tqdm(data_paths):
        print(f"Test dataset: {data_path}")
        kpi = data_path.split('/')[-1][:-4]

        dataloader, labels, raw_series = prepare_data(data_path)
        recons = test_a_dataset(model=model, dataloader=dataloader)


        y_series = raw_series[-len(recons):]
        y_labels = labels[-len(recons):]
        y_scores = minmax_scale(np.abs(recons - y_series))

        f1_range = best_f1_score_range(y_labels, y_scores)
        f1_point = best_f1_score_point(y_labels, y_scores)

        res.append([f1_range, f1_point])

    return res

# %%
data_dir = "/root/data/train/"
data_paths = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if not i.startswith(".")]
samples = sample(data_paths, 7)

pre_train_paths, fine_tune_paths = samples[:5], samples[5:]

def prepare_data(data_path, shuffle=False):
    data_df = pd.read_csv(data_path)
    raw_series = data_df['value'].to_numpy()

    raw_series = minmax_scale(raw_series)

    dataset = KAD_DisformerTestSet(raw_series, win_len=20, seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, drop_last=False, shuffle=shuffle)

    return dataloader, dataset

his_datasets = [prepare_data(i, shuffle=True)[1] for i in pre_train_paths]
his_dataset = ConcatDataset(his_datasets)

his_dataloader = DataLoader(his_dataset, batch_size=256, drop_last=True, shuffle=True)
test_dataloader1 = prepare_data(fine_tune_paths[0], shuffle=False)[0]
test_dataloader2 = prepare_data(fine_tune_paths[1], shuffle=False)[0]


model = KAD_Disformer(N=0, d_model=20, layers=1)
print("Pre-train result:")
print(test(model, fine_tune_paths))

print("Fine-tune results:")
fine_tune(model, his_dataloader, test_dataloader1)
print(test(model, fine_tune_paths[:1]))

model.load_state_dict(torch.load("./saved_models/KAD_Disformer_sample.ptm"))
fine_tune(model, his_dataloader, test_dataloader2)
print(test(model, fine_tune_paths[1:]))




