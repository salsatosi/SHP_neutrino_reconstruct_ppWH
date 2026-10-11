import pandas as pd
import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import datetime
import joblib
import json
import torch.nn.functional as F


ACTIVATION_MAP = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "elu": F.elu,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

DEFAULT_HYPERPARAMETERS = {
    "random_seed": 42,
    "data_split": {
        "train_val_size": 0.3,
        "random_state": 42
    },
    "flow": {
        "n_layers": 1,
        "hidden_features": 128,
        "num_bins": 32,
        "tails": "linear",
        "tail_bound": 5.0,
        "num_blocks": 2,
        "dropout_probability": 0.15,
        "activation": "silu",
        "min_bin_width": 0.01,
        "min_bin_height": 0.01,
        "min_derivative": 0.01
    },
    "training": {
        "learning_rate": 3e-4,
        "target_noise_std": 0.02,
        "epochs": 200,
        "train_batch_size": 512,
        "validation_batch_size": 1024,
        "test_batch_size": 512,
        "train_shuffle": True,
        "validation_shuffle": False,
        "test_shuffle": False,
        "weight_decay": 1e-4,
        "max_grad_norm": 5.0
    },
    "scheduler": {
        "type": "cosine",
        "eta_min": 1e-6
    }
}


def load_hyperparameters(filename=None):
    if filename is None:
        return DEFAULT_HYPERPARAMETERS.copy()

    with open(filename, "r") as f:
        return json.load(f)


def save_hyperparameters(hyperparameters, filename="hyperparameters"):
    output_filename = f"models/{filename}_nflow_{time_str}.json"
    with open(output_filename, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    return output_filename


FEATURES = ['ChargedLep_Charge',
            "ChargedLep_E",
            "ChargedLep_Eta",
            "ChargedLep_pT",
            "HT",
            "Higgs_E",
            "Higgs_eta",
            "Higgs_pT",
            'LeadBJet_E',
            'LeadBJet_Eta',
            'LeadBJet_pT',
            "MET",
            "MET_Sig",
            'SubLeadBJet_E',
            'SubLeadBJet_Eta',
            'SubLeadBJet_pT',
            "NLightJets",
            "SumLightJets_E",
            "SumLightJets_Pt",
            "SumLightJets_eta",
            'Relative_Phi_Lep_Higgs',
            'Relative_Phi_Lep_MET',
            'Relative_Phi_Lep_LeadBJet',
            'Relative_Phi_Lep_SubleadBJet',
            'Relative_Phi_Lep_SumLightJet']

# FEATURES = ["ChargedLep_E",
#             "ChargedLep_Eta",
#             "ChargedLep_pT",
#             "HT",
#             "Higgs_E",
#             "Higgs_eta",
#             "Higgs_pT",
#             "MET",
#             "MET_Sig",
#             "NLightJets",
#             "SumLightJets_E",
#             "SumLightJets_Pt",
#             "SumLightJets_eta",
#             'Relative_Phi_Lep_Higgs',
#             'Relative_Phi_Lep_MET',
#             'Relative_Phi_Lep_Jet']

TARGET = ["True_nu_eta"]


def read_file(filename,FEATURES,TARGET):
    """
    Read a csv file and return a pandas dataframe.
    :param filename:
    :param FEATURES: list of input features
    :param TARGET: list of target features
    :return:
    """
    df = pd.read_csv(filename,encoding_errors='ignore')


    for col in FEATURES[-5:]:
        df[col] = np.arctan2(np.sin(df[col]),np.cos(df[col]))

    features_data = df[FEATURES].values
    target_data = df[TARGET].values

    return features_data, target_data


def split_data(features, target, hyperparameters):
    """
    Split data into training, validation and testing sets (70%/15%/15%).
    :param features: features dataset
    :param target: target dataset
    :param hyperparameters: hyperparameters
    :return:
    """
    features_train, features_val, target_train, target_val = train_test_split(
        features,
        target,
        test_size=hyperparameters["data_split"]["train_val_size"],
        random_state=hyperparameters["data_split"]["random_state"])

    # Fit Scalers
    scaler_features = StandardScaler()

    features_train = scaler_features.fit_transform(features_train)
    features_val = scaler_features.transform(features_val)

    joblib.dump(scaler_features, f"scalers/scaler_features_{time_str}.joblib")

    features_train_tensor = torch.tensor(features_train,dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val,dtype=torch.float32)


    target_train_tensor = torch.tensor(target_train,dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val,dtype=torch.float32)


    train_dataset = TensorDataset(features_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(features_val_tensor, target_val_tensor)



    return train_dataset, val_dataset


def normalizing_flow(features,context_features,hyperparameters):
    """
    Build a normalizing flow model.
    :param features: number of output features
    :param context_features: number of input features
    :param hyperparameters: hyperparameters
    :return:
    """
    base_gaussian = StandardNormal(shape=[features])
    transforms = []
    act_str = hyperparameters["flow"].get("activation", "relu")
    activation_fn = ACTIVATION_MAP.get(act_str.lower(), F.relu)

    for _ in range(hyperparameters["flow"]["n_layers"]):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=hyperparameters["flow"]["hidden_features"],
                context_features=context_features,
                num_bins=hyperparameters["flow"]["num_bins"],
                tails=hyperparameters["flow"]["tails"],
                tail_bound=hyperparameters["flow"]["tail_bound"],
                num_blocks=hyperparameters["flow"]["num_blocks"],
                dropout_probability=hyperparameters["flow"]["dropout_probability"],
                activation=activation_fn,
                min_bin_width=hyperparameters["flow"]["min_bin_width"],
                min_bin_height=hyperparameters["flow"]["min_bin_height"],
                min_derivative=hyperparameters["flow"]["min_derivative"],
            )
        )

    transform = CompositeTransform(transforms)
    flow = Flow(transform=transform, distribution=base_gaussian)
    flow.to(device)
    return flow


def train_model(flow,train_dataset,val_dataset,hyperparameters):

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["training"]["train_batch_size"],
        shuffle=hyperparameters["training"]["train_shuffle"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["training"]["validation_batch_size"],
        shuffle=hyperparameters["training"]["validation_shuffle"])

    best_val_loss = float("inf")
    optimizer = optim.AdamW(
        flow.parameters(),
        lr=hyperparameters["training"]["learning_rate"],
        weight_decay=hyperparameters["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hyperparameters["training"]["epochs"],
        eta_min=hyperparameters["scheduler"]["eta_min"]
    )

    for epoch in range(hyperparameters["training"]["epochs"]):
        flow.train()
        total_train_loss = 0
        batch_idx = 0
        for batch_features, batch_targets in train_loader:
            if flow.training and hyperparameters["training"]["target_noise_std"] > 0:
                noise = torch.randn_like(batch_targets) * hyperparameters["training"]["target_noise_std"]
                batch_targets = batch_targets + noise

            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            log_prob = flow.log_prob(inputs=batch_targets, context=batch_features)
            loss = -torch.mean(log_prob)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                flow.parameters(),
                max_norm=hyperparameters["training"]["max_grad_norm"]
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
        scheduler.step()

        print(
            f"Epoch [{epoch + 1:02d}/{hyperparameters['training']['epochs']}] | "
            f"Train NLL: {avg_train_loss:.4f} | "
            f"Val NLL: {avg_val_loss:.4f} | "
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(flow.state_dict(), f"models/model_nflow_{time_str}.pt")
            print(f"  --> Saved new best model with Val NLL: {best_val_loss:.4f}")


def test_model(flow,test_dataset,hyperparameters):
    flow.load_state_dict(torch.load(f"models/model_nflow_{time_str}.pt"))
    flow.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparameters["training"]["test_batch_size"],
        shuffle=hyperparameters["training"]["test_shuffle"])

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
    hyperparameter_file = "config_nflow.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)
    torch.manual_seed(hyperparameters["random_seed"])
    save_hyperparameters(hyperparameters)
    print('Hyperparameters saved.')

    n_features = len(FEATURES)
    n_target = len(TARGET)
    features, target = read_file("data/new_Input_WH_NeutrinoEtaFindingForTraining_4th_October2026.csv", FEATURES, TARGET)
    train_dataset, val_dataset = split_data(features, target, hyperparameters)
    flow = normalizing_flow(
        features=n_target,
        context_features=n_features,
        hyperparameters=hyperparameters)
    train_model(
        flow=flow,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hyperparameters=hyperparameters)
