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
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
###############################################################################
# Global Hyperparameters
###############################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.3
RANDOM_SEED = 42

# GAN hyperparameters
LATENT_DIM = 64
GAN_BATCH_SIZE = 64
GAN_LR = 1e-4
GAN_EPOCHS = 20  # Number of epochs for training DCGAN or the improved DCGAN

# When performing oversampling on the minority class, the number of generated samples
# can be set to `len(X_minority)*OVERSAMPLE_RATIO`,
# or set manually to match the size of the majority class, etc.
OVERSAMPLE_RATIO = 1.0

# Some hyperparameters in the improved DCGAN (WGAN-GP)
N_CRITIC = 5         # Number of times the discriminator is trained per epoch
LAMBDA_GP = 10       # Gradient penalty coefficient
HA_COEFF = 0.001     # Additional loss coefficient (for example, Historical Averaging)
HA_DECAY = 0.99      # Decay factor for historical averaging

# XGBoost training configuration
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}

# Whether to enable XGBoost's scale_pos_weight
USE_SCALE_POS_WEIGHT = True

# Top_k features for feature selection
TOP_K_FEATURES = 10

# Directory for saving figures
FIG_SAVE_DIR = "figures"

###############################################################################
# 1. Load and preprocess data
###############################################################################
def load_and_preprocess_data(csv_path, test_size=0.3, random_state=42):
    """
    Load and preprocess a given CSV file for binary classification:
      - In the 'Class' column: 'Normal' -> 0, everything else -> 1
      - One-hot encode protocol_type, service, flag
      - Standardize features
      - Return X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # (1) Convert "Class" to binary
    df["Label"] = df["Class"].apply(lambda x: 0 if str(x).lower()=="normal" else 1)

    # (2) Drop Class, subclass, and other unnecessary columns
    for col in ["Class", "subclass"]:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)

    # (3) One-hot encoding
    cat_cols = ["protocol_type", "service", "flag"]
    for c in cat_cols:
        if c in df.columns:
            df = pd.get_dummies(df, columns=[c], drop_first=True)

    # (4) Remove inf, NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # (5) Split features and labels
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values
    y = df["Label"].values

    # (6) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


###############################################################################
# 2. Define DCGAN Generator / Discriminator (traditional)
###############################################################################
class DCGAN_Generator(nn.Module):
    """
    Generator: Two/three fully-connected layers + BN + ReLU
    """
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class DCGAN_Discriminator(nn.Module):
    """
    Discriminator: Multi-layer fully-connected + LeakyReLU + single logit output
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # Output a single logit
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_dcgan(
    X_minority,
    latent_dim=64,
    num_epochs=20,
    batch_size=64,
    lr=1e-4,
    device="cpu"
):
    """
    Train DCGAN (BCE Loss) on minority class data and return the generator,
    discriminator, and the training loss logs
    """
    ds = TensorDataset(torch.tensor(X_minority, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_minority.shape[1]
    G = DCGAN_Generator(latent_dim, d_dim).to(device)
    D = DCGAN_Discriminator(d_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # 1) Train the discriminator
            optD.zero_grad()
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            # Discriminator loss on real samples
            d_out_real = D(real_data)
            d_loss_real = criterion(d_out_real, real_labels)

            # Discriminator loss on generated samples
            z = torch.randn(bsz, latent_dim, device=device)
            fake_data = G(z)
            d_out_fake = D(fake_data.detach())
            d_loss_fake = criterion(d_out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optD.step()

            # 2) Train the generator
            optG.zero_grad()
            d_out_fake_for_g = D(fake_data)  # no detach
            g_loss = criterion(d_out_fake_for_g, real_labels)
            g_loss.backward()
            optG.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        G_losses.append(g_loss_epoch / len(loader))
        D_losses.append(d_loss_epoch / len(loader))
        print(f"[DCGAN] Epoch {epoch+1}/{num_epochs}, G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    return G, D, G_losses, D_losses


###############################################################################
# 3. Improved DCGAN: Use WGAN-GP loss + additional penalty (example)
###############################################################################
class WGAN_Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class WGAN_Discriminator(nn.Module):
    """
    This can also be called a "Critic". We do not use Sigmoid or BCE here.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic, real_data, fake_data, device="cpu"):
    """
    Compute the gradient penalty for WGAN-GP
    """
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def train_wgan_gp(
    X_minority,
    latent_dim=64,
    num_epochs=20,
    batch_size=64,
    lr=1e-4,
    device="cpu",
    n_critic=5,
    lambda_gp=10,
    ha_coeff=0.001,
    ha_decay=0.99
):
    """
    Improved DCGAN (WGAN-GP + Historical Averaging as an additional penalty)
    - WGAN-GP: Uses the Wasserstein distance + Gradient Penalty
    - Adds an additional loss term in both G and D:
      ha_coeff * sum( (theta - theta_hist)^2 )
    """
    ds = TensorDataset(torch.tensor(X_minority, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_minority.shape[1]
    G = WGAN_Generator(latent_dim, d_dim).to(device)
    D = WGAN_Discriminator(d_dim).to(device)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Historical averaging parameters
    hist_G = [p.clone().detach() for p in G.parameters()]
    hist_D = [p.clone().detach() for p in D.parameters()]

    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # (1) Train the discriminator (critic) multiple times
            for _ in range(n_critic):
                optD.zero_grad()

                # Generate fake samples
                z = torch.randn(bsz, latent_dim, device=device)
                fake_data = G(z)

                d_real = D(real_data)
                d_fake = D(fake_data.detach())
                wdist = d_real.mean() - d_fake.mean()  # WGAN objective
                d_loss_base = - wdist  # We want to maximize wdist => minimize -wdist

                # Compute gradient penalty
                gp = gradient_penalty(D, real_data, fake_data, device=device)

                # Total discriminator loss
                d_loss_total = d_loss_base + lambda_gp * gp

                # (Additional) Historical Averaging
                ha_loss_d = 0.0
                for p, hp in zip(D.parameters(), hist_D):
                    ha_loss_d += ((p - hp) ** 2).mean()
                d_loss_total = d_loss_total + ha_coeff * ha_loss_d

                d_loss_total.backward()
                optD.step()

                # Update historical parameters
                for i_p, p in enumerate(D.parameters()):
                    hist_D[i_p] = ha_decay * hist_D[i_p] + (1 - ha_decay) * p.detach()

            # (2) Train the generator once
            optG.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_data = G(z)
            d_fake_for_g = D(fake_data)
            g_loss_base = - d_fake_for_g.mean()  # WGAN: maximize E[d_fake] => minimize -E[d_fake]

            ha_loss_g = 0.0
            for p, hp in zip(G.parameters(), hist_G):
                ha_loss_g += ((p - hp) ** 2).mean()
            g_loss_total = g_loss_base + ha_coeff * ha_loss_g

            g_loss_total.backward()
            optG.step()

            # Update G's historical parameters
            for i_p, p in enumerate(G.parameters()):
                hist_G[i_p] = ha_decay * hist_G[i_p] + (1 - ha_decay) * p.detach()

            d_loss_epoch += d_loss_total.item()
            g_loss_epoch += g_loss_total.item()

        D_losses.append(d_loss_epoch / len(loader))
        G_losses.append(g_loss_epoch / len(loader))
        print(f"[WGAN-GP] Epoch {epoch+1}/{num_epochs}, G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    return G, D, G_losses, D_losses


###############################################################################
# 4. Evaluation Metrics
###############################################################################
def evaluate_metrics(y_true, y_pred, y_score=None):
    """
    y_true, y_pred (0/1 array), y_score (continuous score [0,1]) => compute:
       - acc+
       - acc-
       - accuracy
       - precision
       - recall
       - F1
       - G-mean
       - AUC
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_pos = tp / (tp + fn + 1e-9)  # acc+
    acc_neg = tn / (tn + fp + 1e-9)  # acc-

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
    Train XGBoost and output metrics. Can also plot and save the ROC curve.
    """
    if USE_SCALE_POS_WEIGHT:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        spw = n_neg / (n_pos + 1e-9)
        XGB_PARAMS["scale_pos_weight"] = spw
    else:
        if "scale_pos_weight" in XGB_PARAMS:
            XGB_PARAMS.pop("scale_pos_weight", None)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    met = evaluate_metrics(y_test, y_pred, y_score)

    print(f"\n--- {method_tag} XGBoost Metrics ---")
    print(f"acc+: {met['acc+']:.4f}")
    print(f"acc-: {met['acc-']:.4f}")
    print(f"accuracy: {met['accuracy']:.4f}")
    print(f"precision: {met['precision']:.4f}")
    print(f"recall: {met['recall']:.4f}")
    print(f"F1: {met['F1']:.4f}")
    print(f"G-mean: {met['G-mean']:.4f}")
    if met["AUC"] is not None:
        print(f"AUC: {met['AUC']:.4f}")

        if plot_roc:
            # Ensure the output directory exists
            os.makedirs(FIG_SAVE_DIR, exist_ok=True)

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
            fname = os.path.join(FIG_SAVE_DIR, f"ROC_{method_tag.replace(' ', '_')}.png")
            plt.savefig(fname)
            plt.close()

    return met, model


###############################################################################
# 5. Feature selection based on permutation entropy / fuzzy entropy + RFE
###############################################################################
from collections import Counter

def permutation_entropy_1d(data, m=3, tau=1):
    data = np.asarray(data)
    N = len(data)
    if N < m * tau:
        return 0.0

    pattern_counter = Counter()
    for i in range(N - (m - 1) * tau):
        window = data[i : i + m * tau : tau]
        pattern = tuple(np.argsort(window))
        pattern_counter[pattern] += 1

    total_count = sum(pattern_counter.values())
    pe = 0.0
    for freq in pattern_counter.values():
        p = freq / total_count
        pe -= p * np.log2(p)
    return pe

def fuzzy_entropy_1d(data, m=2, r=0.2):
    data = np.asarray(data)
    N = len(data)
    if N < m + 1:
        return 0.0

    X_m = np.array([data[i : i + m] for i in range(N - m + 1)])
    X_m1 = np.array([data[i : i + m + 1] for i in range(N - (m + 1) + 1)])

    def _phi(X_block):
        count_list = []
        for i in range(len(X_block)):
            dist = np.abs(X_block - X_block[i]).max(axis=1)
            count_list.append(np.sum(np.exp(- (dist ** 2) / r)))
        return np.sum(count_list) / (len(X_block) * (len(X_block) - 1))

    phi_m = _phi(X_m)
    phi_m1 = _phi(X_m1)
    if phi_m1 == 0:
        return 0.0
    return np.log(phi_m / phi_m1 + 1e-9)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def compute_entropies_and_select_features(X, y, feature_names,
                                          top_k=10, max_samples=1000,
                                          method="permutation"):
    np.random.seed(RANDOM_SEED)
    N = X.shape[0]
    if N > max_samples:
        idx_sub = np.random.choice(N, max_samples, replace=False)
        X_sub = X[idx_sub]
    else:
        X_sub = X

    n_features = X.shape[1]
    ent_list = []
    if method == "permutation":
        print("[*] Using Permutation Entropy")
        for i in range(n_features):
            ent_i = permutation_entropy_1d(X_sub[:, i], m=3, tau=1)
            ent_list.append(ent_i)
    elif method == "fuzzy":
        print("[*] Using Fuzzy Entropy")
        for i in range(n_features):
            ent_i = fuzzy_entropy_1d(X_sub[:, i], m=2, r=0.2)
            ent_list.append(ent_i)
    else:
        raise ValueError("method must be 'permutation' or 'fuzzy'")

    print("\n=== Entropy Values ===")
    for name, v in zip(feature_names, ent_list):
        print(f"{name}: {v:.4f}")

    base_est = LogisticRegression(max_iter=1000)
    selector = RFE(base_est, n_features_to_select=top_k)
    selector.fit(X, y)
    rfe_mask = selector.support_

    idx_by_entropy = np.argsort(ent_list)[::-1]
    entropy_mask = np.zeros(n_features, dtype=bool)
    entropy_mask[idx_by_entropy[:top_k]] = True

    final_mask = rfe_mask & entropy_mask
    if not np.any(final_mask):
        print("[!] Intersection is empty => fallback to union")
        final_mask = rfe_mask | entropy_mask

    sel_indices = np.where(final_mask)[0]
    sel_features = [feature_names[i] for i in sel_indices]

    print(f"\nRFE selected: {rfe_mask.sum()} features; Entropy top {top_k}; final => {len(sel_indices)} features.")
    print("Final selected features:", sel_features)

    return sel_indices, sel_features


###############################################################################
# 6. Perform feature selection directly based on XGBoost importance
###############################################################################
def xgb_feature_selection(X, y, feature_names, top_k=10):
    if USE_SCALE_POS_WEIGHT:
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        spw = n_neg / (n_pos + 1e-9)
        XGB_PARAMS["scale_pos_weight"] = spw
    else:
        if "scale_pos_weight" in XGB_PARAMS:
            XGB_PARAMS.pop("scale_pos_weight", None)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    sel_indices = idx_sorted[:top_k]
    sel_features = [feature_names[i] for i in sel_indices]

    print("\n=== XGBoost Feature Importance ===")
    for i in idx_sorted:
        print(f"{feature_names[i]} => {importances[i]:.4f}")

    print(f"\nSelect top {top_k} features => {sel_features}")
    return sel_indices, sel_features


###############################################################################
# 7. Full pipeline for a single dataset
###############################################################################
def run_experiment_for_dataset(csv_path):
    """
    (1) Read the original data, train XGBoost, output metrics + plot ROC
    (2) Oversample using DCGAN -> train XGBoost
    (3) Oversample using improved DCGAN (WGAN-GP + additional penalty) -> train XGBoost
    (4) Feature selection (based on permutation/fuzzy entropy + RFE), then repeat steps (1)-(3)
    (5) Feature selection (based on XGBoost importance), then repeat steps (1)-(3)
    """
    print(f"\n===== Processing dataset: {csv_path} =====")
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data(
        csv_path,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    print(f"Train size = {len(X_train)}, Test size = {len(X_test)}")
    print(f"Positive in train = {sum(y_train == 1)}, Negative in train = {sum(y_train == 0)}")

    #*****************************
    # (1) XGBoost on the original data
    #*****************************
    met_original, model_org = train_xgboost_and_evaluate(
        X_train, y_train, X_test, y_test,
        plot_roc=True,
        method_tag="Original"
    )

    #-----------------------------
    # Separate minority / majority classes
    #-----------------------------
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]

    if len(X_min) == 0:
        print("[!] No minority samples => skip oversampling steps.")
        return

    #*****************************
    # (2) DCGAN oversampling
    #*****************************
    print("\n--- Training DCGAN for minority oversampling ---")
    G_dc, D_dc, G_loss_dc, D_loss_dc = train_dcgan(
        X_minority=X_min,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE
    )
    # Ensure the output directory exists
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    # Plot the DCGAN loss
    plt.figure()
    plt.plot(G_loss_dc, label="G_loss")
    plt.plot(D_loss_dc, label="D_loss")
    plt.title("DCGAN Losses")
    plt.legend()
    # Save figure to FIG_SAVE_DIR
    dcgan_loss_path = os.path.join(FIG_SAVE_DIR, "DCGAN_Loss.png")
    plt.savefig(dcgan_loss_path)
    plt.close()

    # Generate new samples
    n_gen = int(len(X_min) * OVERSAMPLE_RATIO)
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    X_min_syn_dc = G_dc(z).detach().cpu().numpy()

    # Rebuild the training set
    X_train_dc = np.vstack([X_maj, X_min, X_min_syn_dc])
    y_train_dc = np.concatenate([
        np.zeros(len(X_maj)),
        np.ones(len(X_min)),
        np.ones(len(X_min_syn_dc))
    ])
    idx_perm = np.random.permutation(len(X_train_dc))
    X_train_dc = X_train_dc[idx_perm]
    y_train_dc = y_train_dc[idx_perm]

    met_dcgan, model_dcgan = train_xgboost_and_evaluate(
        X_train_dc, y_train_dc, X_test, y_test,
        plot_roc=True,
        method_tag="DCGAN Oversampled"
    )

    #*****************************
    # (3) Improved DCGAN (WGAN-GP+HA, etc.)
    #*****************************
    print("\n--- Training WGAN-GP (improved DCGAN) for minority oversampling ---")
    G_wg, D_wg, G_loss_wg, D_loss_wg = train_wgan_gp(
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
    # Plot the WGAN-GP losses
    plt.figure()
    plt.plot(G_loss_wg, label="G_loss")
    plt.plot(D_loss_wg, label="D_loss")
    plt.title("WGAN-GP+extra Losses")
    plt.legend()
    wgan_loss_path = os.path.join(FIG_SAVE_DIR, "WGAN_GP_Loss.png")
    plt.savefig(wgan_loss_path)
    plt.close()

    # Generate new samples
    n_gen = int(len(X_min) * OVERSAMPLE_RATIO)
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    X_min_syn_wg = G_wg(z).detach().cpu().numpy()

    # Rebuild the training set
    X_train_wg = np.vstack([X_maj, X_min, X_min_syn_wg])
    y_train_wg = np.concatenate([
        np.zeros(len(X_maj)),
        np.ones(len(X_min)),
        np.ones(len(X_min_syn_wg))
    ])
    idx_perm = np.random.permutation(len(X_train_wg))
    X_train_wg = X_train_wg[idx_perm]
    y_train_wg = y_train_wg[idx_perm]

    met_wgan, model_wgan = train_xgboost_and_evaluate(
        X_train_wg, y_train_wg, X_test, y_test,
        plot_roc=True,
        method_tag="WGAN-GP Oversampled"
    )

    #****************************************************************
    # (4) Feature selection (based on permutation/fuzzy entropy + RFE), repeat (1)-(3)
    #****************************************************************
    print("\n=== Feature Selection: Entropy + RFE ===")
    sel_indices, sel_features = compute_entropies_and_select_features(
        X_train, y_train, feat_names,
        top_k=TOP_K_FEATURES,
        max_samples=500,
        method="permutation"   # or "fuzzy"
    )

    # Build the subset based on selected features
    X_train_sel = X_train[:, sel_indices]
    X_test_sel = X_test[:, sel_indices]

    # (a) Original data subset
    met_orig_fs, _ = train_xgboost_and_evaluate(
        X_train_sel, y_train, X_test_sel, y_test,
        plot_roc=True,
        method_tag="Original+FS"
    )

    # (b) DCGAN oversampling + FS
    X_maj_sel = X_maj[:, sel_indices]
    X_min_sel = X_min[:, sel_indices]

    # Retrain DCGAN specifically for these selected features
    G_dc2, _, G_loss2, D_loss2 = train_dcgan(
        X_minority=X_min_sel,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE
    )
    # Generate
    n_gen2 = int(len(X_min_sel) * OVERSAMPLE_RATIO)
    z2 = torch.randn(n_gen2, LATENT_DIM, device=DEVICE)
    X_min_syn_dc2 = G_dc2(z2).detach().cpu().numpy()

    # Rebuild
    X_train_dc2 = np.vstack([X_maj_sel, X_min_sel, X_min_syn_dc2])
    y_train_dc2 = np.concatenate([
        np.zeros(len(X_maj_sel)),
        np.ones(len(X_min_sel)),
        np.ones(len(X_min_syn_dc2))
    ])
    idx_perm = np.random.permutation(len(X_train_dc2))
    X_train_dc2 = X_train_dc2[idx_perm]
    y_train_dc2 = y_train_dc2[idx_perm]

    met_dcgan_fs, _ = train_xgboost_and_evaluate(
        X_train_dc2, y_train_dc2, X_test_sel, y_test,
        plot_roc=True,
        method_tag="DCGAN+FS"
    )

    # (c) Improved DCGAN (WGAN-GP) + FS
    G_wg2, _, G_loss3, D_loss3 = train_wgan_gp(
        X_minority=X_min_sel,
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
    n_gen3 = int(len(X_min_sel) * OVERSAMPLE_RATIO)
    z3 = torch.randn(n_gen3, LATENT_DIM, device=DEVICE)
    X_min_syn_wg2 = G_wg2(z3).detach().cpu().numpy()

    X_train_wg2 = np.vstack([X_maj_sel, X_min_sel, X_min_syn_wg2])
    y_train_wg2 = np.concatenate([
        np.zeros(len(X_maj_sel)),
        np.ones(len(X_min_sel)),
        np.ones(len(X_min_syn_wg2))
    ])
    idx_perm = np.random.permutation(len(X_train_wg2))
    X_train_wg2 = X_train_wg2[idx_perm]
    y_train_wg2 = y_train_wg2[idx_perm]

    met_wgan_fs, _ = train_xgboost_and_evaluate(
        X_train_wg2, y_train_wg2, X_test_sel, y_test,
        plot_roc=True,
        method_tag="WGAN-GP+FS"
    )

    #****************************************************************
    # (5) Feature selection based on XGBoost importance => repeat (1)-(3)
    #****************************************************************
    print("\n=== Feature Selection: XGBoost importance ===")
    sel_indices2, sel_features2 = xgb_feature_selection(
        X_train, y_train, feat_names, top_k=TOP_K_FEATURES
    )
    X_train_sel2 = X_train[:, sel_indices2]
    X_test_sel2 = X_test[:, sel_indices2]

    # (a) Original data
    met_orig_fs2, _ = train_xgboost_and_evaluate(
        X_train_sel2, y_train, X_test_sel2, y_test,
        plot_roc=True,
        method_tag="Original+XGB_FS"
    )

    # (b) DCGAN + XGB_FS
    X_min_sel2 = X_min[:, sel_indices2]
    X_maj_sel2 = X_maj[:, sel_indices2]

    G_dc3, _, G_loss4, D_loss4 = train_dcgan(
        X_minority=X_min_sel2,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE
    )
    n_gen4 = int(len(X_min_sel2) * OVERSAMPLE_RATIO)
    z4 = torch.randn(n_gen4, LATENT_DIM, device=DEVICE)
    X_min_syn_dc3 = G_dc3(z4).detach().cpu().numpy()

    X_train_dc3 = np.vstack([X_maj_sel2, X_min_sel2, X_min_syn_dc3])
    y_train_dc3 = np.concatenate([
        np.zeros(len(X_maj_sel2)),
        np.ones(len(X_min_sel2)),
        np.ones(len(X_min_syn_dc3))
    ])
    idx_perm = np.random.permutation(len(X_train_dc3))
    X_train_dc3 = X_train_dc3[idx_perm]
    y_train_dc3 = y_train_dc3[idx_perm]

    met_dcgan_fs2, _ = train_xgboost_and_evaluate(
        X_train_dc3, y_train_dc3, X_test_sel2, y_test,
        plot_roc=True,
        method_tag="DCGAN+XGB_FS"
    )

    # (c) Improved DCGAN + XGB_FS
    G_wg3, _, G_loss5, D_loss5 = train_wgan_gp(
        X_minority=X_min_sel2,
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
    n_gen5 = int(len(X_min_sel2) * OVERSAMPLE_RATIO)
    z5 = torch.randn(n_gen5, LATENT_DIM, device=DEVICE)
    X_min_syn_wg3 = G_wg3(z5).detach().cpu().numpy()

    X_train_wg3 = np.vstack([X_maj_sel2, X_min_sel2, X_min_syn_wg3])
    y_train_wg3 = np.concatenate([
        np.zeros(len(X_maj_sel2)),
        np.ones(len(X_min_sel2)),
        np.ones(len(X_min_syn_wg3))
    ])
    idx_perm = np.random.permutation(len(X_train_wg3))
    X_train_wg3 = X_train_wg3[idx_perm]
    y_train_wg3 = y_train_wg3[idx_perm]

    met_wgan_fs2, _ = train_xgboost_and_evaluate(
        X_train_wg3, y_train_wg3, X_test_sel2, y_test,
        plot_roc=True,
        method_tag="WGAN-GP+XGB_FS"
    )

    print("==== Done ====")


###############################################################################
# 8. Main function
###############################################################################
def main():

    # Ensure the figure directory exists
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)

    data_dir = "NSL"
    data_files = ["combined_data_with_class.csv"]
    data_paths = [os.path.join(data_dir, f) for f in data_files]

    print("[Device in use]", DEVICE)

    for csv_path in data_paths:
        run_experiment_for_dataset(csv_path)

    print("\nAll Done.")


if __name__ == "__main__":
    main()
