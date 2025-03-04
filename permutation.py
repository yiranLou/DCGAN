import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import xgboost as xgb
import matplotlib.pyplot as plt

################################################################################
# Global Hyperparameters
################################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.3
RANDOM_SEED = 42

# GAN hyperparameters
LATENT_DIM = 64
GAN_BATCH_SIZE = 64
GAN_LR = 1e-4
GAN_EPOCHS = 20
N_CRITIC = 5
LAMBDA_GP = 10
HA_COEFF = 0.001
HA_DECAY = 0.99

# Oversampling ratio for the minority class
OVERSAMPLE_RATIO = 1

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}
USE_SCALE_POS_WEIGHT = True  # Whether to use scale_pos_weight in XGBoost


################################################################################
# 1. Data loading and preprocessing
################################################################################
def load_and_preprocess_data(csv_path, test_size=0.3, random_state=42):
    """
    Load the CSV data, clean and preprocess it.
    Expected to have a column "Label" indicating benign or attack (0 or 1).
    """
    df = pd.read_csv(csv_path)
    # Clean the data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"CSV {csv_path} does not contain the column 'Label'")

    # Convert the label
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).lower() == "benign" else 1)

    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values
    y = df["Label"].values

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


################################################################################
# 2. Define Generator and Discriminator (MLP-based)
################################################################################
class DeeperGenerator(nn.Module):
    """
    A 4-layer fully-connected network with BatchNorm + ReLU for table data generation.
    """
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class DeeperDiscriminator(nn.Module):
    """
    A 4-layer fully-connected network with LeakyReLU for the discriminator.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # For WGAN, no Sigmoid
        )

    def forward(self, x):
        return self.net(x).view(-1)


################################################################################
# 3. Train DCGAN (using BCE Loss)
################################################################################
def train_dcgan(X_minority, latent_dim, num_epochs, batch_size, lr, device):
    """
    Train a DCGAN with BCE Loss on the minority samples.
    """
    ds = TensorDataset(torch.tensor(X_minority, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_minority.shape[1]
    G = DeeperGenerator(latent_dim, d_dim).to(device)
    D = DeeperDiscriminator(d_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses, D_losses = [], []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # ---- Train Discriminator ----
            optD.zero_grad()
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            d_out_real = D(real_data)
            d_loss_real = criterion(d_out_real, real_labels)

            z = torch.randn(bsz, latent_dim, device=device)
            fake_data = G(z)
            d_out_fake = D(fake_data.detach())
            d_loss_fake = criterion(d_out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optD.step()
            d_loss_epoch += d_loss.item()

            # ---- Train Generator ----
            optG.zero_grad()
            d_out_fake_for_g = D(fake_data)
            g_loss = criterion(d_out_fake_for_g, real_labels)
            g_loss.backward()
            optG.step()
            g_loss_epoch += g_loss.item()

        G_losses.append(g_loss_epoch / len(loader))
        D_losses.append(d_loss_epoch / len(loader))
        print(f"[DCGAN] Epoch {epoch+1}/{num_epochs}, "
              f"G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    # Plot & save
    plt.figure()
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.title("DCGAN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("dcgan_loss_curve.png")
    plt.close()

    return G, D, G_losses, D_losses


################################################################################
# 4. Train WGAN-GP + Historical Averaging
################################################################################
def gradient_penalty(critic, real_data, fake_data, device):
    """
    Calculate gradient penalty for WGAN-GP.
    """
    alpha = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def train_wgan_gp_ha(X_minority, latent_dim, num_epochs, batch_size, lr,
                     device, n_critic=5, lambda_gp=10, ha_coeff=0.001, ha_decay=0.99):
    """
    Train WGAN-GP with Historical Averaging for the minority samples.
    """
    ds = TensorDataset(torch.tensor(X_minority, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_minority.shape[1]
    G = DeeperGenerator(latent_dim, d_dim).to(device)
    D = DeeperDiscriminator(d_dim).to(device)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Historical averaging parameters
    hist_G = [p.clone().detach() for p in G.parameters()]
    hist_D = [p.clone().detach() for p in D.parameters()]

    G_losses, D_losses = [], []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # ---- Train the Discriminator multiple times ----
            for _ in range(n_critic):
                optD.zero_grad()
                z = torch.randn(bsz, latent_dim, device=device)
                fake_data = G(z)

                d_real = D(real_data)
                d_fake = D(fake_data.detach())
                wdist = torch.mean(d_real) - torch.mean(d_fake)
                d_loss_basic = -wdist  # WGAN objective

                gp = gradient_penalty(D, real_data, fake_data, device)
                d_loss_total = d_loss_basic + lambda_gp * gp

                # Historical Averaging for D
                ha_loss_D = 0.0
                for p, hp in zip(D.parameters(), hist_D):
                    ha_loss_D += torch.mean((p - hp)**2)
                d_loss_total += ha_coeff * ha_loss_D

                d_loss_total.backward()
                optD.step()

                # Update historical parameters of D
                for idx_p, p in enumerate(D.parameters()):
                    hist_D[idx_p] = ha_decay * hist_D[idx_p] + (1 - ha_decay) * p.clone().detach()

            # ---- Train the Generator ----
            optG.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_data = G(z)
            d_fake_for_g = D(fake_data)
            g_loss_basic = -torch.mean(d_fake_for_g)

            # Historical Averaging for G
            ha_loss_G = 0.0
            for p, hp in zip(G.parameters(), hist_G):
                ha_loss_G += torch.mean((p - hp)**2)

            g_loss_total = g_loss_basic + ha_coeff * ha_loss_G
            g_loss_total.backward()
            optG.step()

            # Update historical parameters of G
            for idx_p, p in enumerate(G.parameters()):
                hist_G[idx_p] = ha_decay * hist_G[idx_p] + (1 - ha_decay) * p.clone().detach()

            d_loss_epoch += d_loss_total.item()
            g_loss_epoch += g_loss_total.item()

        D_losses.append(d_loss_epoch / len(loader))
        G_losses.append(g_loss_epoch / len(loader))
        print(f"[WGAN-GP+HA] Epoch {epoch+1}/{num_epochs}, "
              f"G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    # Plot & save
    plt.figure()
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.title("WGAN-GP+HA Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("wgangp_ha_loss_curve.png")
    plt.close()

    return G, D, G_losses, D_losses


################################################################################
# 5. Train XGBoost + Evaluate multiple metrics
################################################################################
def evaluate_metrics(y_true, y_pred, y_score=None):
    """
    Compute various metrics: acc+, acc-, accuracy, precision, recall, F1, G-mean, AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_pos = tp / (tp + fn + 1e-9)  # True positive accuracy
    acc_neg = tn / (tn + fp + 1e-9)  # True negative accuracy
    f1 = f1_score(y_true, y_pred, zero_division=0)
    g_mean = np.sqrt(acc_pos * acc_neg)
    roc_auc = None
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

    return {
        "acc+": acc_pos,
        "acc-": acc_neg,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "F1": f1,
        "G-mean": g_mean,
        "AUC": roc_auc
    }


def train_xgboost_and_evaluate(X_train, y_train, X_test, y_test,
                               plot_roc=True, method_tag=""):
    """
    Train XGBoost and evaluate on the test set with multiple metrics.
    Optionally plot the ROC curve.
    """
    if USE_SCALE_POS_WEIGHT:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        spw = n_neg / max(n_pos, 1)
        XGB_PARAMS["scale_pos_weight"] = spw
    else:
        if "scale_pos_weight" in XGB_PARAMS:
            XGB_PARAMS.pop("scale_pos_weight", None)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    met = evaluate_metrics(y_test, y_pred, y_score)
    print(f"\n--- {method_tag} XGB Metrics ---")
    print(f"acc+:      {met['acc+']:.4f}")
    print(f"acc-:      {met['acc-']:.4f}")
    print(f"accuracy:  {met['accuracy']:.4f}")
    print(f"precision: {met['precision']:.4f}")
    print(f"recall:    {met['recall']:.4f}")
    print(f"F1-score:  {met['F1']:.4f}")
    print(f"G-mean:    {met['G-mean']:.4f}")
    if met["AUC"] is not None:
        print(f"AUC:       {met['AUC']:.4f}")

    if plot_roc and met["AUC"] is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{method_tag} (AUC={met['AUC']:.4f})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {method_tag}")
        plt.legend()
        filename_safe_tag = method_tag.replace(" ", "_")
        plt.savefig(f"roc_curve_{filename_safe_tag}.png")
        plt.close()

    return met


################################################################################
# 6. Permutation Entropy + RFE Feature Selection with a Random Subset
################################################################################
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from collections import Counter

def permutation_entropy_1d(data, m=3, tau=1):
    """
    Calculate Permutation Entropy (PE) for a 1D sequence.
    m: embedding dimension
    tau: time delay
    """
    data = np.asarray(data)
    N = len(data)
    if N < m * tau:
        return 0.0

    # Count the frequencies of all permutation patterns
    pattern_counter = Counter()
    for i in range(N - (m - 1)*tau):
        # Extract the window with step = tau
        window = data[i : i + m*tau : tau]
        # Determine the permutation pattern by ranking
        pattern = tuple(np.argsort(window))
        pattern_counter[pattern] += 1

    total_count = sum(pattern_counter.values())
    pe = 0.0
    for freq in pattern_counter.values():
        p = freq / total_count
        pe -= p * np.log2(p)
    return pe


def permutation_entropy_of_vector(vec, m=3, tau=1):
    """
    Wrapper function to compute the Permutation Entropy for a 1D vector.
    """
    return permutation_entropy_1d(vec, m=m, tau=tau)


def compute_entropies_and_select_features(X, y, feature_names, top_k=10, max_pe_samples=500):
    np.random.seed(RANDOM_SEED)

    N = X.shape[0]
    if N > max_pe_samples:
        subset_indices = np.random.choice(N, max_pe_samples, replace=False)
        X_sub = X[subset_indices]
    else:
        X_sub = X

    # 1) Calculate PE for each feature
    n_features = X_sub.shape[1]
    entropy_list = []
    print("\n=== Permutation Entropy for each feature (subset-based) ===")
    for i in range(n_features):
        pe_i = permutation_entropy_of_vector(X_sub[:, i], m=3, tau=1)
        entropy_list.append(pe_i)
        print(f"{feature_names[i]}: {pe_i:.4f}")

    # 2) RFE on the full data
    base_est = LogisticRegression(max_iter=1000)
    selector = RFE(base_est, n_features_to_select=top_k, step=1)
    selector.fit(X, y)
    mask_rfe = selector.support_

    # 3) Combine with top-k by Permutation Entropy
    sorted_indices_by_pe = np.argsort(entropy_list)[::-1]  # descending
    mask_pe = np.zeros_like(mask_rfe, dtype=bool)
    mask_pe[sorted_indices_by_pe[:top_k]] = True

    final_mask = mask_rfe & mask_pe

    # 检查如果交集为空，fallback 改为并集
    if not np.any(final_mask):
        print("Intersection is empty, fallback to union.")
        final_mask = mask_rfe | mask_pe

    selected_indices = np.where(final_mask)[0]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    print(f"\n=> RFE selected: {sum(mask_rfe)} features, "
          f"PE top {top_k}: {top_k} features, "
          f"Combined final: {len(selected_indices)} features.")
    print("Final Selected Features:", selected_feature_names)

    # <== 确保函数在任何情况下都返回一个元组
    return selected_indices, selected_feature_names



################################################################################
# 7. Feature selection via XGBoost importance
################################################################################
def xgboost_feature_selection(X, y, feature_names, top_k=10):
    """
    Train XGBoost on all features, then select top_k features by importance.
    """
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:top_k]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    print("\n=== XGBoost Feature Importances ===")
    for idx in indices:
        print(f"{feature_names[idx]}: importance={importances[idx]:.4f}")

    print(f"\nSelect top {top_k} features: {selected_feature_names}")
    return selected_indices, selected_feature_names


################################################################################
# 8. Main procedure for a single dataset
################################################################################
def run_full_experiment_for_dataset(csv_path,
                                    do_feature_select=False,
                                    select_method="entropy_rfe",
                                    top_k=10):
    """
    For one CSV dataset, do the following:
      A) Original data -> XGBoost
      B) DCGAN oversampling -> XGBoost
      C) WGAN-GP+HA oversampling -> XGBoost
    If do_feature_select=True, apply feature selection on the training set first.
    select_method == "entropy_rfe" 时将使用 Permutation Entropy + RFE (现已替换)。
    select_method == "xgboost" 时使用 XGBoost feature importance。
    """
    print(f"\n===== Processing dataset: {csv_path} =====")
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data(
        csv_path, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Feature selection if needed
    if do_feature_select:
        if select_method == "entropy_rfe":
            # 注意: 这里虽然名字仍是 'entropy_rfe'，
            # 但内部已替换为 'Permutation Entropy' 进行度量
            sel_indices, sel_features = compute_entropies_and_select_features(
                X_train, y_train, feat_names, top_k=top_k, max_pe_samples=500
            )
        elif select_method == "xgboost":
            sel_indices, sel_features = xgboost_feature_selection(
                X_train, y_train, feat_names, top_k=top_k
            )
        else:
            raise ValueError(f"Unknown select_method: {select_method}")

        # Filter
        X_train = X_train[:, sel_indices]
        X_test = X_test[:, sel_indices]
        feat_names = sel_features

    # Separate minority and majority class
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]
    print(f"Majority class training samples: {len(X_maj)}, "
          f"Minority class training samples: {len(X_min)}")

    if len(X_min) == 0:
        print("No minority samples in this dataset. Only do XGBoost on the original data.")
        _ = train_xgboost_and_evaluate(
            X_train, y_train, X_test, y_test,
            plot_roc=True,
            method_tag="Original-NoMinority"
        )
        return

    # A) XGBoost on original data
    _ = train_xgboost_and_evaluate(
        X_train, y_train, X_test, y_test,
        plot_roc=True,
        method_tag="Original Data"
    )

    # B) DCGAN oversampling
    print(f"\n--- DCGAN Oversampling (Epoch={GAN_EPOCHS}) ---")
    G_dc, D_dc, G_loss_dc, D_loss_dc = train_dcgan(
        X_minority=X_min,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE
    )

    n_gen = int(len(X_min) * OVERSAMPLE_RATIO)
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    X_min_syn_dc = G_dc(z).detach().cpu().numpy()

    X_train_dc = np.vstack([X_maj, X_min, X_min_syn_dc])
    y_train_dc = np.concatenate([
        np.zeros(len(X_maj)),
        np.ones(len(X_min)),
        np.ones(len(X_min_syn_dc))
    ])
    idx_perm = np.random.permutation(len(X_train_dc))
    X_train_dc = X_train_dc[idx_perm]
    y_train_dc = y_train_dc[idx_perm]

    _ = train_xgboost_and_evaluate(
        X_train_dc, y_train_dc, X_test, y_test,
        plot_roc=True,
        method_tag="DCGAN Oversampling"
    )

    # C) WGAN-GP+HA oversampling
    print(f"\n--- WGAN-GP+HA Oversampling (Epoch={GAN_EPOCHS}) ---")
    G_wg, D_wg, G_loss_wg, D_loss_wg = train_wgan_gp_ha(
        X_minority=X_min,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        ha_coeff=HA_COEFF,
        ha_decay=HA_DECAY
    )

    n_gen = int(len(X_min) * OVERSAMPLE_RATIO)
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    X_min_syn_wg = G_wg(z).detach().cpu().numpy()

    X_train_wg = np.vstack([X_maj, X_min, X_min_syn_wg])
    y_train_wg = np.concatenate([
        np.zeros(len(X_maj)),
        np.ones(len(X_min)),
        np.ones(len(X_min_syn_wg))
    ])
    idx_perm = np.random.permutation(len(X_train_wg))
    X_train_wg = X_train_wg[idx_perm]
    y_train_wg = y_train_wg[idx_perm]

    _ = train_xgboost_and_evaluate(
        X_train_wg, y_train_wg, X_test, y_test,
        plot_roc=True,
        method_tag="WGAN-GP+HA Oversampling"
    )

    print(f"===> Finished processing dataset {os.path.basename(csv_path)}.")


################################################################################
# 9. Main function to process multiple datasets
################################################################################
def main():
    data_dir = "data"
    data_files = [
        "D1.csv", "D2.csv", "D3.csv", "D4.csv",
        "D5.csv", "D6.csv", "D7.csv", "D8.csv",
    ]
    data_paths = [os.path.join(data_dir, f) for f in data_files]

    print("[Device in use]", DEVICE)
    for idx, csv_path in enumerate(data_paths, start=1):
        # 1) Without feature selection
        run_full_experiment_for_dataset(
            csv_path,
            do_feature_select=False
        )

        # 2) Permutation Entropy + RFE (替换了原先的Sample Entropy)
        run_full_experiment_for_dataset(
            csv_path,
            do_feature_select=True,
            select_method="entropy_rfe",
            top_k=10
        )

        # 3) XGBoost-based feature selection
        run_full_experiment_for_dataset(
            csv_path,
            do_feature_select=True,
            select_method="xgboost",
            top_k=10
        )

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
