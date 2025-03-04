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
from collections import Counter

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

USE_SCALE_POS_WEIGHT = True  # 二分类中常用
# XGBoost 默认参数（针对二分类）
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "use_label_encoder": False,
    "objective": "binary:logistic",  # 二分类
    "eval_metric": "logloss",        # 二分类
}

################################################################################
# 1. Data loading and preprocessing (仅二分类)
################################################################################
def load_and_preprocess_data(csv_path, label_name="Class",
                             test_size=0.3, random_state=42):
    """
    加载 CSV 数据，并将字符串列(如protocol_type等)编码，然后对 Label 列做二分类映射：
      - Label=0 if 是normal
      - Label=1 if 非normal（攻击）
    """
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()

    # 若数据中有类似 protocol_type / service / flag 等字符串特征，做简单编码
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    if label_name not in df.columns:
        raise ValueError(f"CSV {csv_path} 中没有列 '{label_name}'")

    # 二分类：只要不是"normal"就当 1
    df["Label"] = df[label_name].apply(lambda x: 0 if str(x).lower() == "normal" else 1)

    # 去掉不需要放进特征的列
    feature_cols = list(df.columns)
    for drop_col in ["Class", "Label"]:
        if drop_col in feature_cols:
            feature_cols.remove(drop_col)

    X = df[feature_cols].values
    y = df["Label"].values

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
    4层全连接 + BatchNorm + ReLU 的生成器
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
    4层全连接 + LeakyReLU 的判别器（WGAN 中不加 Sigmoid）
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
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


################################################################################
# 3. Train DCGAN (using BCE Loss)
################################################################################
def train_dcgan(X_minority, latent_dim, num_epochs, batch_size, lr, device):
    """
    用BCE Loss训练DCGAN
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

    # 画损失曲线
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
    训练WGAN-GP+Historical Averaging
    """
    ds = TensorDataset(torch.tensor(X_minority, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_dim = X_minority.shape[1]
    G = DeeperGenerator(latent_dim, d_dim).to(device)
    D = DeeperDiscriminator(d_dim).to(device)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # 历史平均的参数记录
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
                d_loss_basic = -wdist

                gp = gradient_penalty(D, real_data, fake_data, device)
                d_loss_total = d_loss_basic + lambda_gp * gp

                # Historical Averaging for D
                ha_loss_D = 0.0
                for p, hp in zip(D.parameters(), hist_D):
                    ha_loss_D += torch.mean((p - hp)**2)
                d_loss_total += ha_coeff * ha_loss_D

                d_loss_total.backward()
                optD.step()

                # 更新D的历史权重
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

            # 更新G的历史权重
            for idx_p, p in enumerate(G.parameters()):
                hist_G[idx_p] = ha_decay * hist_G[idx_p] + (1 - ha_decay) * p.clone().detach()

            d_loss_epoch += d_loss_total.item()
            g_loss_epoch += g_loss_total.item()

        D_losses.append(d_loss_epoch / len(loader))
        G_losses.append(g_loss_epoch / len(loader))
        print(f"[WGAN-GP+HA] Epoch {epoch+1}/{num_epochs}, "
              f"G_loss={G_losses[-1]:.4f}, D_loss={D_losses[-1]:.4f}")

    # 画损失曲线
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
# 5. Train XGBoost + Evaluate metrics (二分类)
################################################################################
def evaluate_metrics_binary(y_true, y_pred, y_score=None):
    """
    二分类各种指标：acc+, acc-, accuracy, precision, recall, F1, G-mean, AUC
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_pos = tp / (tp + fn + 1e-9)
    acc_neg = tn / (tn + fp + 1e-9)
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
    针对二分类训练XGBoost并评估
    """
    # scale_pos_weight
    if USE_SCALE_POS_WEIGHT:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        spw = n_neg / max(n_pos, 1)
        XGB_PARAMS["scale_pos_weight"] = spw
    else:
        if "scale_pos_weight" in XGB_PARAMS:
            del XGB_PARAMS["scale_pos_weight"]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    met = evaluate_metrics_binary(y_test, y_pred, y_score)
    print(f"\n--- {method_tag} (Binary) XGB Metrics ---")
    print(f"acc+:      {met['acc+']:.4f}")
    print(f"acc-:      {met['acc-']:.4f}")
    print(f"accuracy:  {met['accuracy']:.4f}")
    print(f"precision: {met['precision']:.4f}")
    print(f"recall:    {met['recall']:.4f}")
    print(f"F1-score:  {met['F1']:.4f}")
    print(f"G-mean:    {met['G-mean']:.4f}")
    if met["AUC"] is not None:
        print(f"AUC:       {met['AUC']:.4f}")

    # 如果需要画ROC
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
# 6. 主流程：只做二分类
################################################################################
def run_full_experiment_for_dataset(csv_path):
    """
    对单个CSV数据集执行完整流程：
      A) 原始数据 -> XGBoost
      B) DCGAN合成少数类 -> XGBoost
      C) WGAN-GP+HA合成少数类 -> XGBoost
    """
    print(f"\n===== Processing dataset: {csv_path} =====")
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data(
        csv_path, label_name="Class", test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # 统计正负样本
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]
    print(f"Majority class (y=0): {len(X_maj)}")
    print(f"Minority class (y=1): {len(X_min)}")

    # 如果少数类太少(甚至为 0)，直接跳过GAN
    if len(X_min) < 2:
        print("少数类样本太少或没有，跳过GAN oversampling，仅用原始数据训练XGBoost。")
        _ = train_xgboost_and_evaluate(
            X_train, y_train, X_test, y_test,
            plot_roc=True,
            method_tag="Original-NoMinority"
        )
        return

    # A) 原始数据训练XGBoost
    train_xgboost_and_evaluate(
        X_train, y_train, X_test, y_test,
        plot_roc=True,
        method_tag="Original Data"
    )

    # B) DCGAN 合成少数类
    print(f"\n--- DCGAN Oversampling (Epoch={GAN_EPOCHS}) ---")
    G_dc, D_dc, _, _ = train_dcgan(
        X_minority=X_min,
        latent_dim=LATENT_DIM,
        num_epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        lr=GAN_LR,
        device=DEVICE
    )

    # 生成同等数量的合成少数样本
    n_gen = int(len(X_min) * OVERSAMPLE_RATIO)
    z = torch.randn(n_gen, LATENT_DIM, device=DEVICE)
    X_min_syn_dc = G_dc(z).detach().cpu().numpy()

    X_train_dc = np.vstack([X_maj, X_min, X_min_syn_dc])
    y_train_dc = np.concatenate([
        np.zeros(len(X_maj)),
        np.ones(len(X_min)),
        np.ones(len(X_min_syn_dc))
    ])
    # 打乱
    idx_perm = np.random.permutation(len(X_train_dc))
    X_train_dc = X_train_dc[idx_perm]
    y_train_dc = y_train_dc[idx_perm]

    train_xgboost_and_evaluate(
        X_train_dc, y_train_dc, X_test, y_test,
        plot_roc=True,
        method_tag="DCGAN Oversampling"
    )

    # C) WGAN-GP+HA 合成少数类
    print(f"\n--- WGAN-GP+HA Oversampling (Epoch={GAN_EPOCHS}) ---")
    G_wg, D_wg, _, _ = train_wgan_gp_ha(
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

    train_xgboost_and_evaluate(
        X_train_wg, y_train_wg, X_test, y_test,
        plot_roc=True,
        method_tag="WGAN-GP+HA Oversampling"
    )

    print(f"===> Finished processing dataset {os.path.basename(csv_path)}.")


################################################################################
# 7. Example usage
################################################################################
if __name__ == "__main__":
    data_path = "NSL/combined_data_with_class.csv"

    print("[Device in use]", DEVICE)
    run_full_experiment_for_dataset(data_path)
