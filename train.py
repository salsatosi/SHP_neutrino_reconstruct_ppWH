import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import datetime
import joblib


# torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
FEATURES = ["ChargedLep_E",
                "ChargedLep_Eta",
                "ChargedLep_pT",
                "HT",
                "Higgs_E",
                "Higgs_eta",
                "Higgs_pT",
                "MET",
                "MET_Sig",
                "NLightJets",
                "SumLightJets_E",
                "SumLightJets_Pt",
                "SumLightJets_eta"]

ANGLE_FEATURES = ["ChargedLep_Phi",
                  "Higgs_phi",
                  "MET_Phi",
                  "SumLightJets_phi"]

TARGET = ["True_nu_eta"]


def read_file(filename,FEATURES,TARGET):
    """
    Read a csv file and return a pandas dataframe.
    :param filename:
    :return:
    """
    df = pd.read_csv(filename,encoding_errors='ignore')

    for col in ANGLE_FEATURES:
        df[f'{col}_sin'] = np.sin(df[col])
        df[f'{col}_cos'] = np.cos(df[col])
        FEATURES.extend([f'{col}_sin',f'{col}_cos'])

    features_data = df[FEATURES].values
    target_data = df[TARGET].values

    return features_data, target_data


def split_data(features, target):
    """
    Split data into training, validation and testing sets (70%/15%/15%).
    :param features: features dataset
    :param target: target dataset
    :return:
    """
    features_train, features_temp, target_train, target_temp = train_test_split(
        features, target, test_size=0.3, random_state=42)

    features_val, features_test, target_val, target_test = train_test_split(
        features_temp, target_temp, test_size=0.5, random_state=42
    )

    # Fit Scalers
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    features_train = scaler_features.fit_transform(features_train)
    features_val = scaler_features.transform(features_val)
    features_test = scaler_features.transform(features_test)
    target_train = scaler_target.fit_transform(target_train)
    target_val = scaler_target.transform(target_val)
    target_test = scaler_target.transform(target_test)

    joblib.dump(scaler_features, f"scalers/scaler_features_{time_str}.joblib")
    joblib.dump(scaler_target, f"scalers/scaler_target_{time_str}.joblib")

    features_train_tensor = torch.tensor(features_train,dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val,dtype=torch.float32)
    features_test_tensor = torch.tensor(features_test,dtype=torch.float32)

    target_train_tensor = torch.tensor(target_train,dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val,dtype=torch.float32)
    target_test_tensor = torch.tensor(target_test,dtype=torch.float32)

    train_dataset = TensorDataset(features_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(features_val_tensor, target_val_tensor)
    test_dataset = TensorDataset(features_test_tensor, target_test_tensor)


    return train_dataset, val_dataset, test_dataset


def normalizing_flow(n_layers,features,hidden_features,context_features):
    """
    Build a normalizing flow model.
    :param n_layers: number of hidden layers
    :param features: number of output features
    :param hidden_features: number of nodes in hidden layers
    :param context_features: number of input features
    :return:
    """
    base_gaussian = StandardNormal(shape=[features])
    transforms = []

    for _ in range(n_layers):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_bins=16,
                tails="linear",
                tail_bound=5.0,
                num_blocks=3,
            )
        )

    transform = CompositeTransform(transforms)
    flow = Flow(transform=transform, distribution=base_gaussian)
    flow.to(device)
    return flow


def train_model(flow,lr,train_dataset,val_dataset,epochs):

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    best_val_loss = float("inf")
    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    for epoch in range(epochs):
        flow.train()
        total_train_loss = 0
        batch_idx = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            log_prob = flow.log_prob(inputs=batch_targets, context=batch_features)
            loss = -torch.mean(log_prob)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                flow.parameters(),
                max_norm=10.0
            )
            if batch_idx == 0:
                print("gradient norm:", grad_norm.item())
                batch_idx += 1

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)


        # validation
        flow.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                log_prob = flow.log_prob(inputs=batch_targets, context=batch_features)
                loss = -torch.mean(log_prob)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Train NLL: {avg_train_loss:.4f} | "
            f"Val NLL: {avg_val_loss:.4f} | "
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(flow.state_dict(), f"models/model_{time_str}.pt")
            print(f"  --> Saved new best model with Val NLL: {best_val_loss:.4f}")


def test_model(flow,test_dataset):
    flow.load_state_dict(torch.load(f"models/model_{time_str}.pt"))
    # flow.load_state_dict(torch.load(f"models/model_20260927-200142.pt"))
    flow.eval()
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    total_test_loss = 0
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            log_prob = flow.log_prob(inputs=batch_targets, context=batch_features)
            loss = -torch.mean(log_prob)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"  --> Test NLL: {avg_test_loss:.4f}")


if __name__ == "__main__":
    n_features = len(FEATURES) + len(ANGLE_FEATURES)*2
    n_target = len(TARGET)
    features, target = read_file("data.csv",FEATURES,TARGET)
    train_dataset, val_dataset, test_dataset = split_data(features, target)
    flow = normalizing_flow(
        n_layers=1,
        features=n_target,
        hidden_features=128,
        context_features=n_features)
    train_model(
        flow=flow,
        lr=1e-4,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50)
    test_model(flow, test_dataset)













