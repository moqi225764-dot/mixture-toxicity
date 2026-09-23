#!/usr/bin/env python
# encoding: utf-8

"""混合毒性预测 Demo：不依赖 dgl/se3_transformer，用简化模型跑通端到端训练并可视化
简化点：
  - 跳过 SE3 图分支（用可学习的组分嵌入代替分子图编码）
  - 保留：指纹分支 + 混合物描述符分支 + 混合浓度分支 + 门控融合 + 标签特异性注意力
  - 生成模拟数据：45 个二元混合物 × 10 个 (时间×终点) 标签
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score, mean_squared_error

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟数据生成
# ============================================================
def generate_synthetic_data(n_samples=120, n_components=2, seed=42):
    rng = np.random.default_rng(seed)

    # 5 种抗生素 + 4 种污染物 = 9 种组分，取二元组合
    antibiotics = ['Ciprofloxacin', 'Erythromycin', 'Sulfamethoxazole', 'Tetracycline', 'Trimethoprim']
    pollutants = ['Copper', 'Zinc', 'Lead', 'Cadmium']
    comp_names = antibiotics + pollutants
    n_comp_types = len(comp_names)

    # 每个组分的指纹（166 维 MACCS-like 二值向量）
    comp_fingerprints = rng.integers(0, 2, size=(n_comp_types, 50)).astype(np.float32)

    # 每个组分的描述符（3 维：CATS2D_04_DP / GATS7s / DISPe）
    comp_descriptors = rng.normal(0, 1, size=(n_comp_types, 3)).astype(np.float32)

    # 生成混合物
    rows = []
    for i in range(n_samples):
        c_a_idx = rng.integers(0, n_comp_types)
        c_b_idx = rng.integers(0, n_comp_types)
        while c_b_idx == c_a_idx:
            c_b_idx = rng.integers(0, n_comp_types)

        c_a = rng.uniform(0.1, 5.0)
        c_b = rng.uniform(0.1, 5.0)
        c_total = c_a + c_b
        time = rng.choice([4, 6, 8, 10, 12])

        # 模拟毒性：早期受 GATS7s/DISPe 主导，后期受 CATS2D_04_DP 主导
        cat_a, gats_a, dis_a = comp_descriptors[c_a_idx]
        cat_b, gats_b, dis_b = comp_descriptors[c_b_idx]
        cat_mix = (cat_a + cat_b) / 2
        gats_mix = (gats_a + gats_b) / 2
        dis_mix = (dis_a + dis_b) / 2

        time_factor = time / 12.0
        # 早期毒性：GATS7s + DISPe 贡献大；后期：CATS2D_04_DP 贡献大
        base_tox = (1 - time_factor) * (0.5 * gats_mix + 0.5 * dis_mix) + \
                   time_factor * cat_mix
        dose_effect = np.log(c_total) * 0.3

        rows.append({
            'smiles_a': f'COMP{c_a_idx}',
            'smiles_b': f'COMP{c_b_idx}',
            'c_a': c_a, 'c_b': c_b, 'c_total': c_total,
            'time': time,
            'comp_a_idx': c_a_idx, 'comp_b_idx': c_b_idx,
            'CATS2D_04_DP': cat_mix, 'GATS7s': gats_mix, 'DISPe': dis_mix,
            'base_tox': base_tox,
            'dose_effect': dose_effect,
            'time_factor': time_factor
        })

    df = pd.DataFrame(rows)

    # 生成 10 个标签：5 时间点 × 2 终点
    label_cols = []
    for t in [4, 6, 8, 10, 12]:
        for endpoint in ['EC30', 'EC50']:
            col = f'{t}h_{endpoint}'
            # EC50 > EC30，后期毒性更强
            endpoint_shift = 0.3 if endpoint == 'EC50' else 0.0
            time_weight = (t / 12.0) * 0.4
            noise = rng.normal(0, 0.15, size=n_samples)
            df[col] = (df['base_tox'] + df['dose_effect'] +
                       endpoint_shift + time_weight + noise)
            label_cols.append(col)

    return df, comp_fingerprints, comp_descriptors, label_cols


def prepare_modal_data(df, comp_fingerprints, label_cols):
    n = len(df)
    n_comp = 2

    # 组分指纹 (n, 2, fp_dim)
    fingers = np.stack([
        np.stack([comp_fingerprints[int(df.iloc[i]['comp_a_idx'])],
                  comp_fingerprints[int(df.iloc[i]['comp_b_idx'])]], axis=0)
        for i in range(n)
    ], axis=0).astype(np.float32)

    # 混合物描述符 (n, 3*3=9)  含升维
    mixdesc_raw = df[['CATS2D_04_DP', 'GATS7s', 'DISPe']].values.astype(np.float32)
    mixdesc = np.concatenate(
        [mixdesc_raw, np.square(mixdesc_raw), np.log1p(np.abs(mixdesc_raw))],
        axis=1
    ).astype(np.float32)

    # 混合浓度向量 (n, v_mix_dim)
    c_a = df['c_a'].values.astype(np.float32).reshape(-1, 1)
    c_b = df['c_b'].values.astype(np.float32).reshape(-1, 1)
    c_total = df['c_total'].values.astype(np.float32).reshape(-1, 1)
    x_a = (c_a / c_total).astype(np.float32)
    x_b = (c_b / c_total).astype(np.float32)
    log_c_a = np.log(np.maximum(c_a, 1e-8)).astype(np.float32)
    log_c_b = np.log(np.maximum(c_b, 1e-8)).astype(np.float32)
    log_c_total = np.log(np.maximum(c_total, 1e-8)).astype(np.float32)
    t_vals = df['time'].values.astype(np.float32).reshape(-1, 1)

    # 时间编码 (8 维)
    def time_enc(t, d=8, max_t=12.0):
        enc = np.zeros((d,), dtype=np.float32)
        for k in range(d):
            denom = max_t ** (2 * k / d)
            enc[k] = np.sin(t / denom) if k % 2 == 0 else np.cos(t / denom)
        return enc
    t_enc = np.stack([time_enc(t[0]) for t in t_vals], axis=0).astype(np.float32)

    conc = np.concatenate([x_a, x_b, c_a, c_b, log_c_a, log_c_b, log_c_total, t_vals, t_enc],
                          axis=1).astype(np.float32)

    # 标签
    labels = df[label_cols].values.astype(np.float32)

    # 标准化
    mixdesc_mean, mixdesc_std = mixdesc.mean(0), mixdesc.std(0) + 1e-8
    mixdesc = (mixdesc - mixdesc_mean) / mixdesc_std
    conc_mean, conc_std = conc.mean(0), conc.std(0) + 1e-8
    conc = (conc - conc_mean) / conc_std
    labels_mean, labels_std = labels.mean(0), labels.std(0) + 1e-8
    labels_norm = (labels - labels_mean) / labels_std

    return fingers, mixdesc, conc, labels_norm, labels, labels_mean, labels_std


# ============================================================
# 2. 简化模型（去掉 SE3 图分支，保留其他模块）
# ============================================================
class SimpleFingerBranch(nn.Module):
    def __init__(self, d_in, hidden, out, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out), nn.ReLU(), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x.float())


class SimpleMixdescBranch(nn.Module):
    def __init__(self, d_in, hidden, out, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out), nn.LayerNorm(out), nn.ReLU(), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x.float())


class SimpleConcBranch(nn.Module):
    def __init__(self, d_in, hidden, out, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out), nn.LayerNorm(out), nn.ReLU(), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x.float())


class SimpleSymmetricAggregate(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.phi = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
    def forward(self, x):
        h = self.rho(x)
        return self.phi(h.sum(dim=1))


class SimpleGatedFusion(nn.Module):
    def __init__(self, emb_dim, n_mod=3, hidden=128, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * n_mod, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_mod)
        )
    def forward(self, *embs):
        concat = torch.cat(embs, dim=1)
        w = F.softmax(self.gate(concat), dim=1)
        stacked = torch.stack(embs, dim=1)
        return (stacked * w.unsqueeze(-1)).sum(dim=1), w


class SimpleLabelAttention(nn.Module):
    def __init__(self, emb_dim, context_dim, num_labels, dropout=0.1):
        super().__init__()
        self.label_query = nn.Parameter(torch.randn(num_labels, emb_dim))
        self.context_proj = nn.Linear(context_dim, emb_dim)
        self.W_Q = nn.Linear(emb_dim, emb_dim)
        self.W_K = nn.Linear(emb_dim, emb_dim)
        self.W_V = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_tokens, context):
        B, n, d = F_tokens.shape
        c = self.context_proj(context)
        q = self.label_query.unsqueeze(0) + c.unsqueeze(1)
        Q, K, V = self.W_Q(q), self.W_K(F_tokens), self.W_V(F_tokens)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (d ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return self.out_proj(out), attn


class SimpleMixtureToxModel(nn.Module):
    def __init__(self, fp_dim, mixdesc_dim, conc_dim, num_labels, emb_dim=64, dropout=0.3):
        super().__init__()
        self.finger_branch = SimpleFingerBranch(fp_dim, 64, 16, dropout)
        self.mixdesc_branch = SimpleMixdescBranch(mixdesc_dim, 32, 16, dropout)
        self.conc_branch = SimpleConcBranch(conc_dim, 32, 32, dropout)

        self.finger_proj = nn.Linear(16, emb_dim)
        self.mixdesc_proj = nn.Linear(16, emb_dim)
        self.conc_proj = nn.Linear(32, emb_dim)

        self.sym_agg = SimpleSymmetricAggregate(emb_dim, emb_dim * 2, dropout)

        self.gated_fusion = SimpleGatedFusion(emb_dim, 3, emb_dim * 2, dropout)

        self.label_attn = SimpleLabelAttention(emb_dim, 32, num_labels, dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(emb_dim // 2, 1)
            ) for _ in range(num_labels)
        ])

    def forward(self, component_fingers, mixdesc, conc, return_attn=False):
        B = conc.shape[0]
        N = component_fingers.shape[1]

        flat_fp = component_fingers.view(B * N, -1)
        fp_emb = self.finger_branch(flat_fp)
        fp_emb = self.finger_proj(fp_emb).view(B, N, -1)
        h_fp = self.sym_agg(fp_emb)

        h_mixdesc = self.mixdesc_proj(self.mixdesc_branch(mixdesc))

        h_mix_raw = self.conc_branch(conc)
        h_mix = self.conc_proj(h_mix_raw)
        context = h_mix_raw

        F_raw = torch.stack([h_fp, h_mixdesc, h_mix], dim=1)
        _, gate_w = self.gated_fusion(h_fp, h_mixdesc, h_mix)
        F_tokens = F_raw * gate_w.unsqueeze(-1)

        label_embs, attn_w = self.label_attn(F_tokens, context)
        outputs = torch.stack(
            [self.heads[l](label_embs[:, l]).squeeze(-1) for l in range(len(self.heads))],
            dim=1
        )
        if return_attn:
            return outputs, attn_w, gate_w
        return outputs


# ============================================================
# 3. 训练与可视化
# ============================================================
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 生成数据
    df, comp_fp, comp_desc, label_cols = generate_synthetic_data(n_samples=200, seed=seed)
    print(f'数据集: {len(df)} 个混合物, {len(label_cols)} 个标签')
    print(f'标签: {label_cols}')

    fingers, mixdesc, conc, labels_norm, labels_raw, l_mean, l_std = prepare_modal_data(
        df, comp_fp, label_cols
    )

    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    def take(idxs):
        i = np.asarray(idxs)
        return (torch.tensor(fingers[i], dtype=torch.float32),
                torch.tensor(mixdesc[i], dtype=torch.float32),
                torch.tensor(conc[i], dtype=torch.float32),
                torch.tensor(labels_norm[i], dtype=torch.float32),
                torch.tensor(labels_raw[i], dtype=torch.float32))

    tr_f, tr_m, tr_c, tr_yn, tr_yr = take(tr_idx)
    va_f, va_m, va_c, va_yn, va_yr = take(va_idx)
    te_f, te_m, te_c, te_yn, te_yr = take(te_idx)

    print(f'训练: {len(tr_idx)}, 验证: {len(va_idx)}, 测试: {len(te_idx)}')

    # 模型
    model = SimpleMixtureToxModel(
        fp_dim=fingers.shape[-1],
        mixdesc_dim=mixdesc.shape[1],
        conc_dim=conc.shape[1],
        num_labels=len(label_cols),
        emb_dim=64,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 训练
    epochs = 150
    batch_size = 16
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        # mini-batch
        perm = torch.randperm(len(tr_idx))
        model.train()
        epoch_loss = 0
        n_batches = 0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s+batch_size]
            if len(b) == 0:
                continue
            optimizer.zero_grad()
            out = model(tr_f[b].to(device), tr_m[b].to(device), tr_c[b].to(device))
            loss = F.mse_loss(out, tr_yn[b].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            va_out = model(va_f.to(device), va_m.to(device), va_c.to(device))
            val_loss = F.mse_loss(va_out, va_yn.to(device)).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f'Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

    model.load_state_dict(best_state)
    model.eval()

    # 测试
    with torch.no_grad():
        te_out_norm, te_attn, te_gate = model(
            te_f.to(device), te_m.to(device), te_c.to(device), return_attn=True
        )
        te_out = te_out_norm.cpu().numpy() * l_std + l_mean
        te_yr_np = te_yr.numpy()
        te_attn = te_attn.cpu().numpy()  # (n_test, num_labels, 3)
        te_gate = te_gate.cpu().numpy()  # (n_test, 3)

    # ============================================================
    # 4. 可视化
    # ============================================================
    os.makedirs('results_vis', exist_ok=True)

    # ---- 图1: Loss 曲线 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='训练 Loss', color='#2196F3', linewidth=1.5)
    ax.plot(val_losses, label='验证 Loss', color='#FF5722', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('训练/验证 Loss 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_vis/01_loss_curve.png', dpi=150)
    plt.close()
    print('保存: results_vis/01_loss_curve.png')

    # ---- 图2: 预测 vs 真实散点图（每个标签）----
    n_labels = len(label_cols)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for l in range(n_labels):
        ax = axes[l]
        true_vals = te_yr_np[:, l]
        pred_vals = te_out[:, l]
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=30, c='#2196F3')
        lim_min = min(true_vals.min(), pred_vals.min())
        lim_max = max(true_vals.max(), pred_vals.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1)
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        ax.set_title(f'{label_cols[l]}\nR²={r2:.3f} RMSE={rmse:.3f}', fontsize=9)
        ax.set_xlabel('真实值', fontsize=8)
        ax.set_ylabel('预测值', fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle('测试集: 预测 vs 真实 (10 个标签)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results_vis/02_pred_vs_true.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('保存: results_vis/02_pred_vs_true.png')

    # ---- 图3: 标签注意力权重热图 ----
    # 平均注意力 (num_labels, 3)  modality: [指纹, 混合物描述符, 混合浓度]
    modality_names = ['分子指纹', '混合物描述符', '混合浓度']
    avg_attn = te_attn.mean(axis=0)  # (num_labels, 3)

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(avg_attn, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(3))
    ax.set_xticklabels(modality_names, fontsize=10)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(label_cols, fontsize=9)
    # 标注数值
    for i in range(n_labels):
        for j in range(3):
            ax.text(j, i, f'{avg_attn[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if avg_attn[i, j] < 0.5 else 'white')
    plt.colorbar(im, ax=ax, label='注意力权重')
    ax.set_title('标签特异性注意力权重\n(行=时间×终点标签, 列=模态)')
    plt.tight_layout()
    plt.savefig('results_vis/03_label_attention_heatmap.png', dpi=150)
    plt.close()
    print('保存: results_vis/03_label_attention_heatmap.png')

    # ---- 图4: 门控融合权重 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_gate = te_gate.mean(axis=0)
    bars = ax.bar(modality_names, avg_gate, color=['#4CAF50', '#FF9800', '#2196F3'], alpha=0.8)
    ax.set_ylabel('门控权重 (softmax)')
    ax.set_title('门控融合: 各模态平均权重')
    for bar, val in zip(bars, avg_gate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(avg_gate) * 1.3)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('results_vis/04_gate_weights.png', dpi=150)
    plt.close()
    print('保存: results_vis/04_gate_weights.png')

    # ---- 图5: 注意力随时间变化 ----
    # 提取每个时间点的 EC50 标签注意力
    time_points = [4, 6, 8, 10, 12]
    ec50_indices = [label_cols.index(f'{t}h_EC50') for t in time_points]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, mod_name in enumerate(modality_names):
        vals = [avg_attn[i, j] for i in ec50_indices]
        ax.plot(time_points, vals, 'o-', label=mod_name, linewidth=2, markersize=8)
    ax.set_xlabel('暴露时间 (小时)')
    ax.set_ylabel('注意力权重')
    ax.set_title('EC50 标签注意力随时间变化\n(验证: 早期指纹/描述符主导, 后期各模态均衡)')
    ax.set_xticks(time_points)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_vis/05_attention_over_time.png', dpi=150)
    plt.close()
    print('保存: results_vis/05_attention_over_time.png')

    # ---- 打印汇总 ----
    print('\n========== 结果汇总 ==========')
    print(f'测试集大小: {len(te_idx)}')
    print(f'\n各标签 R²/RMSE:')
    for l in range(n_labels):
        r2 = r2_score(te_yr_np[:, l], te_out[:, l])
        rmse = np.sqrt(mean_squared_error(te_yr_np[:, l], te_out[:, l]))
        print(f'  {label_cols[l]:12s}: R²={r2:.3f}  RMSE={rmse:.3f}')

    print(f'\n门控融合平均权重:')
    for j, name in enumerate(modality_names):
        print(f'  {name}: {avg_gate[j]:.3f}')

    print(f'\nEC50 标签注意力 (指纹/描述符/浓度):')
    for i, t in zip(ec50_indices, time_points):
        print(f'  {t}h_EC50: 指纹={avg_attn[i,0]:.3f} 描述符={avg_attn[i,1]:.3f} 浓度={avg_attn[i,2]:.3f}')

    print('\n可视化图保存在 results_vis/ 目录')
    print('完成。')


if __name__ == '__main__':
    main()
