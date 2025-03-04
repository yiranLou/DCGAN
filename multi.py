import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

# Prevent issues with displaying graphics in certain environments (e.g., headless servers)
matplotlib.use("Agg")

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
GAN_EPOCHS = 20

# If a class has fewer than this many samples, skip DCGAN/WGAN and just replicate
MIN_SAMPLES_FOR_DCGAN = 2

# WGAN-GP improvements
N_CRITIC = 5
LAMBDA_GP = 10
HA_COEFF = 0.001
HA_DECAY = 0.99

# XGBoost training config
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "use_label_encoder": False,   # XGBoost >= 1.0
    "eval_metric": "mlogloss",    # for multi-class
    "objective": "multi:softprob"
}

TOP_K_FEATURES = 10
FIG_SAVE_DIR = "figures"


###############################################################################
# 1. Load data & preprocess
###############################################################################
def load_and_preprocess_data(csv_path, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Drop the old 'Class' column if present
    if "Class" in df.columns:
        df.drop(["Class"], axis=1, inplace=True)

    #=== Modification (A): Merge extremely rare subclasses ===
    # First count the samples of each subclass; merge those with fewer than 5 samples into "rare"
    min_class_count = 5  # You can define the threshold yourself
    subclass_counts = df["subclass"].value_counts()
    rare_list = subclass_counts[subclass_counts < min_class_count].index

    # Rename all subclasses in rare_list to "rare"
    df["subclass"] = df["subclass"].where(~df["subclass"].isin(rare_list), other="rare")
    #=== End of modification (A) ===

    # Encode 'subclass' as multi-class label
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["subclass"])
    df.drop(["subclass"], axis=1, inplace=True)

    # One-hot encode
    cat_cols = ["protocol_type", "service", "flag"]
    for c in cat_cols:
        if c in df.columns:
            df = pd.get_dummies(df, columns=[c], drop_first=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values
    y = df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols, label_encoder


###############################################################################
# 2. Define DCGAN (same as before)
###############################################################################
class DCGAN_Generator(nn.Module):
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


def train_dcgan(
    X_class,
    latent_dim=64,
    num_epochs=20,
    batch_size=64,
    lr=1e-4,
    device="cpu",
    plot_filename=None
):
    ds = TensorDataset(torch.tensor(X_class, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_class.shape[1]
    G = DCGAN_Generator(latent_dim, d_dim).to(device)
    D = DCGAN_Discriminator(d_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses, D_losses = [], []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # 1) Update Discriminator
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

            # 2) Update Generator
            optG.zero_grad()
            d_out_fake_for_g = D(fake_data)
            g_loss = criterion(d_out_fake_for_g, real_labels)  # want 1
            g_loss.backward()
            optG.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        G_losses.append(g_loss_epoch / len(loader))
        D_losses.append(d_loss_epoch / len(loader))
        print(f"[DCGAN] Epoch {epoch+1}/{num_epochs}, "
              f"G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    if plot_filename:
        plt.figure()
        plt.plot(G_losses, label="G_loss")
        plt.plot(D_losses, label="D_loss")
        plt.title("DCGAN Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(plot_filename, dpi=120)
        plt.close()

    return G, D, G_losses, D_losses


###############################################################################
# 3. WGAN-GP (same as before)
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
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha*real_data + (1-alpha)*fake_data
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    penalty = ((grads.norm(2, dim=1) - 1)**2).mean()
    return penalty


def train_wgan_gp(
    X_class,
    latent_dim=64,
    num_epochs=20,
    batch_size=64,
    lr=1e-4,
    device="cpu",
    n_critic=5,
    lambda_gp=10,
    ha_coeff=0.001,
    ha_decay=0.99,
    plot_filename=None
):
    ds = TensorDataset(torch.tensor(X_class, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_class.shape[1]
    G = WGAN_Generator(latent_dim, d_dim).to(device)
    D = WGAN_Discriminator(d_dim).to(device)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    hist_G = [p.clone().detach() for p in G.parameters()]
    hist_D = [p.clone().detach() for p in D.parameters()]

    G_losses, D_losses = [], []

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        for real_data, in loader:
            bsz = real_data.size(0)
            real_data = real_data.to(device)

            # 1) Critic multiple times
            for _ in range(n_critic):
                optD.zero_grad()
                z = torch.randn(bsz, latent_dim, device=device)
                fake_data = G(z)

                d_real = D(real_data).mean()
                d_fake = D(fake_data.detach()).mean()
                wdist = d_real - d_fake
                d_loss_base = -wdist
                gp = gradient_penalty(D, real_data, fake_data, device=device)
                d_loss_total = d_loss_base + lambda_gp*gp

                # Historical averaging
                ha_loss_d = 0.0
                for p,hp in zip(D.parameters(), hist_D):
                    ha_loss_d += ((p - hp)**2).mean()
                d_loss_total += ha_coeff * ha_loss_d

                d_loss_total.backward()
                optD.step()

                for i_p, p in enumerate(D.parameters()):
                    hist_D[i_p] = ha_decay*hist_D[i_p] + (1-ha_decay)*p.detach()

            # 2) Generator
            optG.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_data = G(z)
            d_fake_for_g = D(fake_data).mean()
            g_loss_base = - d_fake_for_g

            ha_loss_g = 0.0
            for p,hp in zip(G.parameters(), hist_G):
                ha_loss_g += ((p - hp)**2).mean()

            g_loss_total = g_loss_base + ha_coeff * ha_loss_g
            g_loss_total.backward()
            optG.step()

            for i_p, p in enumerate(G.parameters()):
                hist_G[i_p] = ha_decay*hist_G[i_p] + (1-ha_decay)*p.detach()

            d_loss_epoch += d_loss_total.item()
            g_loss_epoch += g_loss_total.item()

        D_losses.append(d_loss_epoch / len(loader))
        G_losses.append(g_loss_epoch / len(loader))
        print(f"[WGAN-GP] Epoch {epoch+1}/{num_epochs}, "
              f"G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    if plot_filename:
        plt.figure()
        plt.plot(G_losses, label="G_loss")
        plt.plot(D_losses, label="D_loss")
        plt.title("WGAN-GP Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(plot_filename, dpi=120)
        plt.close()

    return G, D, G_losses, D_losses


###############################################################################
# 4. Multi-class evaluation (same as before)
###############################################################################
def evaluate_multiclass(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Accuracy for each class
    class_accs = []
    for i in range(n_classes):
        class_count = cm[i].sum()
        if class_count == 0:
            class_accs.append(1.0)
        else:
            class_accs.append(cm[i,i] / class_count)

    g_mean = np.prod(class_accs)**(1.0/n_classes)

    cr = classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "classwise_accuracy": class_accs,
        "g_mean": g_mean
    }


def plot_multiclass_roc(
    y_true,
    y_score,
    n_classes,
    method_tag="",
    save_dir="figures"
):
    from sklearn.preprocessing import label_binarize
    # Binarize output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}

    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes

    fpr_dict["macro"], tpr_dict["macro"] = all_fpr, mean_tpr
    roc_auc_dict["macro"] = auc(all_fpr, mean_tpr)

    plt.figure()
    for i in range(n_classes):
        plt.plot(
            fpr_dict[i], tpr_dict[i],
            label=f"Class {i} (AUC={roc_auc_dict[i]:.2f})",
            alpha=0.7
        )
    plt.plot(
        fpr_dict["macro"],
        tpr_dict["macro"],
        label=f"Macro-average (AUC={roc_auc_dict['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=2
    )
    plt.plot([0,1],[0,1],"k--",alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-class ROC - {method_tag}")
    plt.legend(loc="lower right")

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, f"ROC_{method_tag.replace(' ','_')}.png")
    plt.savefig(outpath, dpi=120)
    plt.close()


def train_xgboost_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes,
    method_tag="",
    save_dir="figures",
    plot_roc=True,
    results_file=None
):
    XGB_PARAMS["num_class"] = num_classes
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    met = evaluate_multiclass(y_test, y_pred)

    print(f"\n--- {method_tag} XGBoost Multi-class Metrics ---")
    print(f"Accuracy: {met['accuracy']:.4f}")
    print(f"Macro-Precision: {met['macro_precision']:.4f}")
    print(f"Macro-Recall: {met['macro_recall']:.4f}")
    print(f"Macro-F1: {met['macro_f1']:.4f}")
    print(f"G-mean: {met['g_mean']:.4f}")
    print("Classwise Accuracy:", met["classwise_accuracy"])
    print("Confusion Matrix:\n", met["confusion_matrix"])
    print("Classification Report:\n", met["classification_report"])

    if plot_roc:
        plot_multiclass_roc(
            y_true=y_test,
            y_score=y_prob,
            n_classes=num_classes,
            method_tag=method_tag,
            save_dir=save_dir
        )

    if results_file:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- {method_tag} ---\n")
            f.write(f"Accuracy: {met['accuracy']:.4f}\n")
            f.write(f"Macro-Precision: {met['macro_precision']:.4f}\n")
            f.write(f"Macro-Recall: {met['macro_recall']:.4f}\n")
            f.write(f"Macro-F1: {met['macro_f1']:.4f}\n")
            f.write(f"G-mean: {met['g_mean']:.4f}\n")
            f.write(f"Classwise Accuracy: {met['classwise_accuracy']}\n")
            f.write(f"Confusion Matrix:\n{met['confusion_matrix']}\n")
            f.write("Classification Report:\n")
            f.write(met["classification_report"] + "\n")
            f.write("-------------------------------------------------\n")

    return met, model


###############################################################################
# 5. Feature selection (same as before)
###############################################################################
from collections import Counter

def permutation_entropy_1d(data, m=3, tau=1):
    data = np.asarray(data)
    N = len(data)
    if N < m*tau:
        return 0.0
    pattern_counter = Counter()
    for i in range(N - (m-1)*tau):
        window = data[i:i+m*tau:tau]
        pattern = tuple(np.argsort(window))
        pattern_counter[pattern]+=1
    total_count = sum(pattern_counter.values())
    pe = 0.0
    for freq in pattern_counter.values():
        p = freq/total_count
        pe -= p*np.log2(p)
    return pe


def fuzzy_entropy_1d(data, m=2, r=0.2):
    data = np.asarray(data)
    N = len(data)
    if N < m+1:
        return 0.0
    X_m  = np.array([data[i:i+m]   for i in range(N-m+1)])
    X_m1 = np.array([data[i:i+m+1] for i in range(N-(m+1)+1)])
    def _phi(X_block):
        count_list = []
        for i in range(len(X_block)):
            dist = np.abs(X_block - X_block[i]).max(axis=1)
            count_list.append(np.sum(np.exp(- (dist**2)/r)))
        return np.sum(count_list)/(len(X_block)*(len(X_block)-1))
    phi_m  = _phi(X_m)
    phi_m1 = _phi(X_m1)
    if phi_m1 == 0:
        return 0.0
    return np.log(phi_m / (phi_m1+1e-9) + 1e-9)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def compute_entropies_and_select_features(
    X,
    y,
    feature_names,
    top_k=10,
    max_samples=1000,
    method="permutation",
    results_file=None
):
    np.random.seed(RANDOM_SEED)
    N = X.shape[0]
    if N > max_samples:
        idx_sub = np.random.choice(N, max_samples, replace=False)
        X_sub = X[idx_sub]
    else:
        X_sub = X

    n_features = X.shape[1]
    ent_list = []
    if method=="permutation":
        print("[*] Using Permutation Entropy")
        for i in range(n_features):
            ent_i = permutation_entropy_1d(X_sub[:,i], m=3, tau=1)
            ent_list.append(ent_i)
    elif method=="fuzzy":
        print("[*] Using Fuzzy Entropy")
        for i in range(n_features):
            ent_i = fuzzy_entropy_1d(X_sub[:,i], m=2, r=0.2)
            ent_list.append(ent_i)
    else:
        raise ValueError("method must be 'permutation' or 'fuzzy'")

    print("\n=== Entropy Values ===")
    for name,v in zip(feature_names, ent_list):
        print(f"{name}: {v:.4f}")

    base_est = LogisticRegression(max_iter=1000, multi_class='auto')
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

    print(f"\nRFE selected: {rfe_mask.sum()} features; "
          f"Entropy top {top_k}; final => {len(sel_indices)} features.")
    print("Final selected features:", sel_features)

    if results_file:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Feature Selection ({method}) ---\n")
            f.write("Entropy Values:\n")
            for name,v in zip(feature_names, ent_list):
                f.write(f"{name}: {v:.4f}\n")

            f.write(f"\nRFE selected: {rfe_mask.sum()} features; "
                    f"Entropy top {top_k}; final => {len(sel_indices)} features.\n")
            f.write("Final selected features: " + str(sel_features) + "\n")
            f.write("-------------------------------------------------\n")

    return sel_indices, sel_features


###############################################################################
# 6. XGBoost-based feature selection (same as before)
###############################################################################
def xgb_feature_selection(
    X,
    y,
    feature_names,
    top_k=10,
    results_file=None
):
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    XGB_PARAMS["num_class"] = n_classes

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

    if results_file:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write("\n--- XGBoost Feature Importance ---\n")
            for i in idx_sorted:
                f.write(f"{feature_names[i]} => {importances[i]:.4f}\n")
            f.write(f"\nSelect top {top_k} features => {sel_features}\n")
            f.write("-------------------------------------------------\n")

    return sel_indices, sel_features


###############################################################################
# 7. Multi-class oversampling with "dynamic" param logic (example)
###############################################################################
def multiclass_oversample_dcgan(X_train, y_train, latent_dim, epochs, batch_size, lr):
    classes = np.unique(y_train)
    counts = [np.sum(y_train == c) for c in classes]
    max_count = np.max(counts)

    X_parts = []
    y_parts = []

    for c in classes:
        X_c = X_train[y_train == c]
        n_c = len(X_c)
        X_parts.append(X_c)
        y_parts.append(np.full(n_c, c))

        if n_c < max_count:
            needed = max_count - n_c
            if n_c < MIN_SAMPLES_FOR_DCGAN:
                print(f"Class {c} has only {n_c} samples => replicate instead of DCGAN.")
                replicate_times = (needed // n_c) + 1
                X_reps = np.tile(X_c, (replicate_times, 1))
                X_reps = X_reps[:needed]
                X_parts.append(X_reps)
                y_parts.append(np.full(X_reps.shape[0], c))

            else:
                #=== Modification (B): Example of dynamic hyperparameter adjustment ===
                # This is just an example; you can make it more complex if needed.
                dynamic_epoch = min(epochs, 10 + int(epochs * (100 / (n_c+1))))
                dynamic_lr = lr
                if n_c < 20:
                    dynamic_lr = lr * 0.5
                print(f"[DCGAN] Class={c}, n_c={n_c}, dynamic_epoch={dynamic_epoch}, dynamic_lr={dynamic_lr}")

                plot_filename = os.path.join(
                    FIG_SAVE_DIR, f"DCGAN_Losses_class_{c}.png"
                )
                G, _, _, _ = train_dcgan(
                    X_c,
                    latent_dim=latent_dim,
                    num_epochs=dynamic_epoch,
                    batch_size=batch_size,
                    lr=dynamic_lr,
                    device=DEVICE,
                    plot_filename=plot_filename
                )
                z = torch.randn(needed, latent_dim, device=DEVICE)
                X_synth = G(z).detach().cpu().numpy()
                X_parts.append(X_synth)
                y_parts.append(np.full(needed, c))

    X_ov = np.vstack(X_parts)
    y_ov = np.concatenate(y_parts)
    idx_perm = np.random.permutation(len(X_ov))
    return X_ov[idx_perm], y_ov[idx_perm]


def multiclass_oversample_wgan(X_train, y_train, latent_dim, epochs, batch_size, lr):
    classes = np.unique(y_train)
    counts = [np.sum(y_train == c) for c in classes]
    max_count = np.max(counts)

    X_parts = []
    y_parts = []

    for c in classes:
        X_c = X_train[y_train == c]
        n_c = len(X_c)
        X_parts.append(X_c)
        y_parts.append(np.full(n_c, c))

        if n_c < max_count:
            needed = max_count - n_c
            if n_c < MIN_SAMPLES_FOR_DCGAN:
                print(f"[WGAN] Class {c} has only {n_c} samples => replicate.")
                replicate_times = (needed // n_c) + 1
                X_reps = np.tile(X_c, (replicate_times, 1))
                X_reps = X_reps[:needed]
                X_parts.append(X_reps)
                y_parts.append(np.full(X_reps.shape[0], c))
            else:
                #=== Modification (B): Example of dynamic hyperparameter adjustment ===
                dynamic_epoch = min(epochs, 10 + int(epochs * (100 / (n_c+1))))
                dynamic_lr = lr
                if n_c < 20:
                    dynamic_lr = lr * 0.5

                print(f"[WGAN] Class={c}, n_c={n_c}, dynamic_epoch={dynamic_epoch}, dynamic_lr={dynamic_lr}")

                plot_filename = os.path.join(
                    FIG_SAVE_DIR, f"WGAN_GP_Losses_class_{c}.png"
                )
                G, _, _, _ = train_wgan_gp(
                    X_c,
                    latent_dim=latent_dim,
                    num_epochs=dynamic_epoch,
                    batch_size=batch_size,
                    lr=dynamic_lr,
                    device=DEVICE,
                    n_critic=N_CRITIC,
                    lambda_gp=LAMBDA_GP,
                    ha_coeff=HA_COEFF,
                    ha_decay=HA_DECAY,
                    plot_filename=plot_filename
                )
                z = torch.randn(needed, latent_dim, device=DEVICE)
                X_synth = G(z).detach().cpu().numpy()
                X_parts.append(X_synth)
                y_parts.append(np.full(needed, c))

    X_ov = np.vstack(X_parts)
    y_ov = np.concatenate(y_parts)
    idx_perm = np.random.permutation(len(X_ov))
    return X_ov[idx_perm], y_ov[idx_perm]


###############################################################################
# 8. Experiment (same as original, just calls the above)
###############################################################################
def run_experiment_for_dataset(csv_path):
    print(f"\n===== Processing dataset: {csv_path} =====")
    X_train, X_test, y_train, y_test, feat_names, label_enc = load_and_preprocess_data(
        csv_path, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    unique_labels = np.unique(y_train)
    n_classes = len(unique_labels)
    print(f"Train size={len(X_train)}, Test size={len(X_test)}, #Classes={n_classes}")
    print("Class distribution in train:",
          {label_enc.inverse_transform([c])[0]: sum(y_train == c) for c in unique_labels})

    results_file = os.path.join(FIG_SAVE_DIR, "results_summary.txt")
    fs_file = os.path.join(FIG_SAVE_DIR, "feature_selection_results.txt")

    # 1) Original data
    met_orig, model_orig = train_xgboost_and_evaluate(
        X_train, y_train, X_test, y_test,
        num_classes=n_classes,
        method_tag="Original",
        results_file=results_file
    )

    # 2) DCGAN oversampling
    print("\n--- Multi-class DCGAN oversampling ---")
    X_train_dc, y_train_dc = multiclass_oversample_dcgan(
        X_train, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_dcgan, _ = train_xgboost_and_evaluate(
        X_train_dc, y_train_dc, X_test, y_test,
        num_classes=n_classes,
        method_tag="DCGAN Oversampled",
        results_file=results_file
    )

    # 3) WGAN-GP
    print("\n--- Multi-class WGAN-GP oversampling ---")
    X_train_wg, y_train_wg = multiclass_oversample_wgan(
        X_train, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_wgan, _ = train_xgboost_and_evaluate(
        X_train_wg, y_train_wg, X_test, y_test,
        num_classes=n_classes,
        method_tag="WGAN-GP Oversampled",
        results_file=results_file
    )

    # 4) Feature selection (Permutation Entropy + RFE)
    print("\n=== Feature Selection: Entropy + RFE ===")
    sel_indices, sel_features = compute_entropies_and_select_features(
        X_train, y_train, feat_names,
        top_k=TOP_K_FEATURES,
        max_samples=500,
        method="permutation",
        results_file=fs_file
    )
    X_train_sel = X_train[:, sel_indices]
    X_test_sel  = X_test[:,  sel_indices]

    met_orig_fs, _ = train_xgboost_and_evaluate(
        X_train_sel, y_train, X_test_sel, y_test,
        num_classes=n_classes,
        method_tag="Original+FS",
        results_file=results_file
    )

    # Re-oversample + FS
    X_train_dc_sel, y_train_dc_sel = multiclass_oversample_dcgan(
        X_train_sel, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_dcgan_fs, _ = train_xgboost_and_evaluate(
        X_train_dc_sel, y_train_dc_sel, X_test_sel, y_test,
        num_classes=n_classes,
        method_tag="DCGAN+FS",
        results_file=results_file
    )

    X_train_wg_sel, y_train_wg_sel = multiclass_oversample_wgan(
        X_train_sel, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_wgan_fs, _ = train_xgboost_and_evaluate(
        X_train_wg_sel, y_train_wg_sel, X_test_sel, y_test,
        num_classes=n_classes,
        method_tag="WGAN-GP+FS",
        results_file=results_file
    )

    # 5) XGBoost-based feature selection
    print("\n=== Feature Selection: XGBoost importance ===")
    sel_indices_xgb, sel_features_xgb = xgb_feature_selection(
        X_train, y_train, feat_names,
        top_k=TOP_K_FEATURES,
        results_file=fs_file
    )
    X_train_sel2 = X_train[:, sel_indices_xgb]
    X_test_sel2  = X_test[:,  sel_indices_xgb]

    met_orig_fs2, _ = train_xgboost_and_evaluate(
        X_train_sel2, y_train, X_test_sel2, y_test,
        num_classes=n_classes,
        method_tag="Original+XGB_FS",
        results_file=results_file
    )

    X_train_dc_sel2, y_train_dc_sel2 = multiclass_oversample_dcgan(
        X_train_sel2, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_dcgan_fs2, _ = train_xgboost_and_evaluate(
        X_train_dc_sel2, y_train_dc_sel2, X_test_sel2, y_test,
        num_classes=n_classes,
        method_tag="DCGAN+XGB_FS",
        results_file=results_file
    )

    X_train_wg_sel2, y_train_wg_sel2 = multiclass_oversample_wgan(
        X_train_sel2, y_train,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR
    )
    met_wgan_fs2, _ = train_xgboost_and_evaluate(
        X_train_wg_sel2, y_train_wg_sel2, X_test_sel2, y_test,
        num_classes=n_classes,
        method_tag="WGAN-GP+XGB_FS",
        results_file=results_file
    )

    print("==== Done ====")


###############################################################################
# 9. Main
###############################################################################
def main():
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
