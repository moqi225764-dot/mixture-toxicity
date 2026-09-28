#!/usr/bin/env python
# encoding: utf-8

"""真实数据版混合毒性预测网站（Streamlit）
启动：
  C:\\Users\\admin\\.conda\\envs\\tabpfn\\python.exe -m streamlit run web_app_real.py
数据：Xu et al. Environ. Health 2024, 5 种抗生素 45 条二元混合射线 × 5 时间 × pEC30/pEC50
模型：saved_models/real_model.pt（全量训练；指标来自按射线分组的 5 折 OOF）
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import streamlit as st

from demo_mixture_tox import SimpleMixtureToxModel
from data.build_real_features import (
    SMILES, component_features, eq4_linear, eq5_rms, build_context,
    DESC_NAMES, FP_BITS
)

HERE = os.path.dirname(os.path.abspath(__file__))

# 云端 Linux 无微软雅黑/黑体：优先注册仓库内置中文字体，保证图表中文不乱码
from matplotlib import font_manager as _fm
_FONT_FILE = os.path.join(HERE, 'fonts', 'SimHei.ttf')
if os.path.exists(_FONT_FILE):
    try:
        _fm.fontManager.addfont(_FONT_FILE)
    except Exception:
        pass
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMES = [4, 6, 8, 10, 12]
LABEL_COLS = [f'{t}h_{e}' for t in TIMES for e in ['EC30', 'EC50']]
MODALITY_NAMES = ['分子指纹', '混合物描述符', '配方/时间上下文']
ABBR_CN = {
    'ENR': '恩诺沙星', 'CTC': '盐酸金霉素', 'TMP': '甲氧苄啶',
    'CMP': '氯霉素', 'ETM': '红霉素'
}
COMBOS = ['ENR-CTC', 'ENR-CMP', 'ENR-ETM', 'CTC-TMP', 'CTC-CMP',
          'CTC-ETM', 'TMP-CMP', 'TMP-ETM', 'CMP-ETM']
RAYS = ['R1', 'R2', 'R3', 'R4', 'R5']


# ------------------------------------------------------------
# 资源加载（缓存）
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    ckpt = torch.load(os.path.join(HERE, 'saved_models', 'real_model.pt'),
                      map_location=DEVICE, weights_only=False)
    dims = ckpt['dims']
    model = SimpleMixtureToxModel(
        fp_dim=dims['fp_dim'], mixdesc_dim=dims['mixdesc_dim'],
        conc_dim=dims['context_dim'], num_labels=dims['num_labels'],
        emb_dim=dims['emb_dim'], dropout=dims['dropout']
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    comp_fp, comp_desc = {}, {}
    for abbr, (smi, cid) in SMILES.items():
        fp, desc, _ = component_features(abbr, smi)
        comp_fp[abbr] = fp
        comp_desc[abbr] = desc

    rays = pd.read_csv(os.path.join(HERE, 'data', 'real_mixture_rays.csv'))
    wide = pd.read_csv(os.path.join(HERE, 'data', 'real_mixtures_wide.csv'))
    comp_info = pd.read_csv(os.path.join(HERE, 'data', 'real_components.csv'))
    return model, ckpt['stats'], ckpt, comp_fp, comp_desc, rays, wide, comp_info


def run_predict(combo, p_a, comp_fp, comp_desc, model, stats):
    """任意摩尔分数的连续预测（特征解析构建，支持射线间插值）。"""
    a, b = combo.split('-')
    p_b = 1.0 - p_a
    p = np.array([p_a, p_b], dtype=np.float64)
    x = np.stack([comp_desc[a], comp_desc[b]], axis=0)

    fingers = np.stack([comp_fp[a], comp_fp[b]], axis=0)[None].astype(np.float32)
    md = np.concatenate([eq4_linear(p, x), eq5_rms(p, x)])[None].astype(np.float32)
    ctx = build_context(float(p_a))[None].astype(np.float32)

    md_n = (md - stats['md_mean']) / stats['md_std']
    ctx_n = (ctx - stats['ctx_mean']) / stats['ctx_std']

    with torch.no_grad():
        Xf = torch.tensor(fingers, device=DEVICE)
        Xm = torch.tensor(md_n, device=DEVICE)
        Xc = torch.tensor(ctx_n, device=DEVICE)
        out, attn, gate = model(Xf, Xm, Xc, return_attn=True)
    pred = (out.cpu().numpy()[0] * stats['l_std'][0] + stats['l_mean'][0])
    return pred, attn.cpu().numpy()[0], gate.cpu().numpy()[0]


# ------------------------------------------------------------
# 图
# ------------------------------------------------------------
def fig_gate(gate):
    colors = ['#4caf50', '#ff9800', '#2196f3']
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    left = 0.0
    for k in range(3):
        ax.barh(0, gate[k], left=left, color=colors[k], edgecolor='white', height=0.5)
        ax.text(left + gate[k] / 2, 0, f'{gate[k]:.3f}', ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')
        left += gate[k]
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis('off')
    ax.legend(MODALITY_NAMES, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False, fontsize=9)
    return fig


def fig_attention(attn):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(attn.T, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=attn.max() * 1.1 + 1e-6)
    ax.set_xticks(range(len(LABEL_COLS)))
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(MODALITY_NAMES, fontsize=9)
    for i in range(3):
        for j in range(len(LABEL_COLS)):
            ax.text(j, i, f'{attn[j, i]:.2f}', ha='center', va='center', fontsize=7,
                    color='black' if attn[j, i] < attn.max() * 0.6 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title('标签特异性注意力：每个时间×终点标签对三模态的关注权重', fontsize=10)
    plt.tight_layout()
    return fig


def fig_prediction_bars(pred, current_time):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = ['#e53935' if TIMES[i // 2] == current_time else '#90caf9'
              for i in range(len(LABEL_COLS))]
    bars = ax.bar(range(len(LABEL_COLS)), pred, color=colors)
    ax.set_xticks(range(len(LABEL_COLS)))
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('预测 pEC（越大越毒）')
    ax.set_title('10 项时间依赖毒性预测（红色=当前查询时间点）')
    for b, v in zip(bars, pred):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=7.5)
    lo, hi = pred.min() - 0.15, pred.max() + 0.15
    ax.set_ylim(lo, hi)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    return fig


def fig_ray_grid(combo, model, stats, comp_fp, comp_desc, wide):
    """5 条真实射线 × 5 时间：预测 vs 实测（pEC30 / pEC50）。"""
    sub = wide[wide['combo'] == combo].sort_values('no')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, ep in enumerate(['EC30', 'EC50']):
        pred_mat = np.zeros((5, 5))
        true_mat = np.zeros((5, 5))
        for ri, (_, r) in enumerate(sub.iterrows()):
            p, _, _ = run_predict(combo, float(r['mole_frac_a']),
                                  comp_fp, comp_desc, model, stats)
            for ti, t in enumerate(TIMES):
                pred_mat[ri, ti] = p[LABEL_COLS.index(f'{t}h_{ep}')]
                true_mat[ri, ti] = r[f'{t}h_{ep}']
        vmin = min(pred_mat.min(), true_mat.min()) - 0.05
        vmax = max(pred_mat.max(), true_mat.max()) + 0.05
        fracs = [f"R{k+1}\n({sub.iloc[k]['mole_frac_a']:.3f}/{sub.iloc[k]['mole_frac_b']:.3f})"
                 for k in range(5)]
        for col, (mat, title) in enumerate([(pred_mat, '模型预测'), (true_mat, '论文实测')]):
            ax = axes[row, col]
            im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
            ax.set_xticks(range(5)); ax.set_xticklabels([f'{t}h' for t in TIMES])
            ax.set_yticks(range(5)); ax.set_yticklabels(fracs, fontsize=8)
            ax.set_title(f'p{ep} {title}')
            for ri in range(5):
                for ti in range(5):
                    ax.text(ti, ri, f'{mat[ri, ti]:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'{combo}（{ABBR_CN[combo.split("-")[0]]} + {ABBR_CN[combo.split("-")[1]]}）'
                 f'5 条射线 × 5 时间 pEC 网格（括号内为摩尔分数 A/B）', fontsize=12)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
st.set_page_config(page_title='真实混合毒性预测系统', page_icon='🧪', layout='wide')
st.markdown("""
<style>
    .main-header {font-size: 28px; font-weight: 800; color: #1a237e; margin-bottom: 0;}
    .sub-header {color: #5c6bc0; font-size: 13px; margin-bottom: 16px;}
    .big-metric {background: #f5f7ff; border-radius: 12px; padding: 16px 18px;
                 text-align: center; border: 1px solid #c5cae9;}
    .big-metric .label {font-size: 13px; color: #5c6bc0;}
    .big-metric .value {font-size: 30px; font-weight: 800; color: #1a237e;}
    .hint {font-size: 12px; color: #78909c;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 抗生素二元混合毒性时间依赖预测（真实数据）</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">数据来源：Xu et al., Environ. Health 2024, '
            '5 种抗生素（ENR/CTC/TMP/CMP/ETM）45 条混合射线 × 4/6/8/10/12h × pEC30/pEC50，'
            '受试生物：青海弧菌 Q67。模型按射线 5 折交叉验证 OOF R²=0.81，'
            '分子特征为 RDKit 真实计算。</div>', unsafe_allow_html=True)

with st.spinner('加载真实模型中...'):
    model, stats, ckpt, comp_fp, comp_desc, df_rays, df_wide, df_comp = load_assets()

page = st.sidebar.radio('功能导航', ['🔍 单案例预测', '🧮 射线网格对照',
                                    '📊 模型性能', '📁 批量预测'])

# ==================== 页面1：单案例预测 ====================
if page == '🔍 单案例预测':
    with st.sidebar:
        st.markdown('### 输入条件')
        combo = st.selectbox('抗生素组合', COMBOS,
                             format_func=lambda c: f"{c}（{ABBR_CN[c.split('-')[0]]}+"
                                                   f"{ABBR_CN[c.split('-')[1]]}）")
        mode = st.radio('混合比例', ['选择真实射线 R1-R5', '自定义摩尔分数（插值）'])
        sub = df_rays[df_rays['combo'] == combo].sort_values('no')
        if mode.startswith('选择'):
            ray = st.select_slider('射线', RAYS, value='R3')
            row = sub[sub['ray'] == ray].iloc[0]
            p_a = float(row['mole_frac_a'])
        else:
            p_a = st.slider('组分 A 摩尔分数 p_A', 0.01, 0.99, 0.5, 0.001)
            # 动态可信度提示
            fracs_sorted = sub.sort_values('mole_frac_a')['mole_frac_a'].values
            r_names = sub.sort_values('mole_frac_a')['ray'].values
            if p_a < min(fracs_sorted) or p_a > max(fracs_sorted):
                level = ('red', '❌ 外推区域',
                         '当前比例超出论文 R1–R5 实验范围，无数据支撑，预测仅供方向性参考。')
            else:
                idx = np.searchsorted(fracs_sorted, p_a)
                if idx <= 1 or idx >= len(fracs_sorted) - 1:
                    level = ('orange', '⚠️ 边缘区域',
                             f'当前比例接近 {r_names[max(0, idx-1)]}（{fracs_sorted[max(0, idx-1)]:.3f}），'
                             f'属边缘预测，内插 R²≈0.72，可信度中等。')
                else:
                    r_lo, r_hi = r_names[idx-1], r_names[idx]
                    f_lo, f_hi = fracs_sorted[idx-1], fracs_sorted[idx]
                    level = ('green', '✅ 高可信度',
                             f'当前比例落在 {r_lo}({f_lo:.3f})–{r_hi}({f_hi:.3f}) 之间，'
                             f'属内插预测，内插 R²≈0.85，可信度高。')
            color_map = {
                'green': ('#2e7d32', '#e8f5e9'),
                'orange': ('#ef6c00', '#fff3e0'),
                'red': ('#c62828', '#ffebee')
            }
            fg, bg = color_map[level[0]]
            st.markdown(
                f'<div style="background:{bg};border-left:4px solid {fg};'
                f'padding:10px 14px;border-radius:6px;margin:8px 0;">'
                f'<span style="color:{fg};font-weight:600;">{level[1]}</span>'
                f'<span style="color:#424242;font-size:13px;margin-left:8px;">{level[2]}</span>'
                f'</div>', unsafe_allow_html=True)
        time_h = st.select_slider('暴露时间 t (h)', TIMES, value=8)

    a, b = combo.split('-')
    p_b = 1.0 - p_a
    pred, attn, gate = run_predict(combo, p_a, comp_fp, comp_desc, model, stats)
    i30 = LABEL_COLS.index(f'{time_h}h_EC30')
    i50 = LABEL_COLS.index(f'{time_h}h_EC50')

    # 若为真实射线，取出实测值对照
    measured = None
    ray_rows = df_wide[(df_wide['combo'] == combo) &
                       (np.isclose(df_wide['mole_frac_a'], p_a))]
    if len(ray_rows):
        measured = ray_rows.iloc[0]

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.0, 1.0, 1.0])
    c1.markdown(f'<div class="big-metric"><div class="label">组分 A</div>'
                f'<div class="value" style="font-size:20px">{ABBR_CN[a]}</div>'
                f'<div class="hint">{a}，p={p_a:.3f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="big-metric"><div class="label">组分 B</div>'
                f'<div class="value" style="font-size:20px">{ABBR_CN[b]}</div>'
                f'<div class="hint">{b}，p={p_b:.3f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="big-metric"><div class="label">暴露时间</div>'
                f'<div class="value">{time_h}h</div></div>', unsafe_allow_html=True)
    txt30 = f'{pred[i30]:.3f}' + (f' <span class="hint">实测 {measured[f"{time_h}h_EC30"]:.3f}</span>'
                                  if measured is not None else '')
    txt50 = f'{pred[i50]:.3f}' + (f' <span class="hint">实测 {measured[f"{time_h}h_EC50"]:.3f}</span>'
                                  if measured is not None else '')
    c4.markdown(f'<div class="big-metric"><div class="label">{time_h}h pEC30 预测</div>'
                f'<div class="value" style="font-size:24px">{txt30}</div></div>',
                unsafe_allow_html=True)
    c5.markdown(f'<div class="big-metric"><div class="label">{time_h}h pEC50 预测</div>'
                f'<div class="value" style="font-size:24px">{txt50}</div></div>',
                unsafe_allow_html=True)

    st.markdown('')
    tab1, tab2, tab3 = st.tabs(['📈 10 项预测', '🎛️ 门控权重', '🎯 标签注意力'])
    with tab1:
        st.pyplot(fig_prediction_bars(pred, time_h), use_container_width=True)
    with tab2:
        st.pyplot(fig_gate(gate), use_container_width=True)
    with tab3:
        st.pyplot(fig_attention(attn), use_container_width=True)

# ==================== 页面2：射线网格对照 ====================
elif page == '🧮 射线网格对照':
    st.markdown('### 5 条真实射线 × 5 个时间点：模型预测 vs 论文实测')
    combo = st.selectbox('抗生素组合', COMBOS, key='grid_combo',
                         format_func=lambda c: f"{c}（{ABBR_CN[c.split('-')[0]]}+"
                                               f"{ABBR_CN[c.split('-')[1]]}）")
    with st.spinner('计算 50 个网格点中...'):
        fig = fig_ray_grid(combo, model, stats, comp_fp, comp_desc, df_wide)
    st.pyplot(fig, use_container_width=True)
    st.caption('纵轴括号为摩尔分数（A/B）。颜色越红 pEC 越大、毒性越强。'
               '左右两图越接近，表示模型对该组合的复现越好。')

# ==================== 页面3：模型性能 ====================
elif page == '📊 模型性能':
    st.markdown('### 模型性能（按射线分组的 5 折交叉验证，45 条样本全部有 OOF 预测）')
    m = ckpt['label_metrics']
    dfm = pd.DataFrame([
        {'标签': k, 'R2': round(v['r2'], 3), 'RMSE(pEC)': round(v['rmse'], 3)}
        for k, v in m.items()
    ])
    c1, c2, c3 = st.columns(3)
    c1.metric('平均 R²', f"{ckpt['mean_r2']:.3f}")
    c2.metric('平均 RMSE', f"{ckpt['mean_rmse']:.3f}")
    c3.metric('论文 QSAR R² 区间', '0.818–0.913')
    st.dataframe(dfm, use_container_width=True, hide_index=True)
    st.caption('注：本文验证为"留出整条射线"（每折测试 9 个组合各 1 条射线），'
               '比论文 80/20 随机划分更严格；4h 指标优于 12h，与论文结论一致。')
    for fname, title in [
        ('01_oof_pred.png', 'OOF 预测 vs 实测（10 标签）'),
        ('02_gate.png', '门控融合各模态权重'),
        ('03_attention.png', '标签×模态注意力热图'),
        ('04_time.png', '注意力随暴露时间变化'),
    ]:
        path = os.path.join(HERE, 'results_vis', 'real', fname)
        st.markdown(f'**{title}**')
        st.image(path, use_container_width=True)

# ==================== 页面4：批量预测 ====================
else:
    st.markdown('### 批量预测')
    st.markdown('上传 CSV（列：`组合, 摩尔分数A, 暴露时间`），组合取值：' +
                '、'.join(COMBOS) + '；时间 4/6/8/10/12。'
                '摩尔分数可填任意值（射线之间按特征解析插值）。')
    tpl = pd.DataFrame([
        {'组合': 'ENR-CMP', '摩尔分数A': 0.346, '暴露时间': 4},
        {'组合': 'CMP-ETM', '摩尔分数A': 0.5, '暴露时间': 12},
    ])
    st.download_button('⬇️ 下载 CSV 模板',
                       tpl.to_csv(index=False).encode('utf-8-sig'),
                       file_name='real_batch_template.csv', mime='text/csv')
    up = st.file_uploader('上传 CSV', type=['csv'])
    if up is not None:
        df_up = pd.read_csv(up)
        need = ['组合', '摩尔分数A', '暴露时间']
        if any(c not in df_up.columns for c in need):
            st.error(f'缺少列，需要：{need}')
        else:
            rows, errs = [], []
            bar = st.progress(0.0)
            for i, r in df_up.iterrows():
                try:
                    combo = str(r['组合']).strip()
                    if combo not in COMBOS:
                        raise ValueError(f'未知组合 {combo}')
                    pa = float(r['摩尔分数A']); t = int(r['暴露时间'])
                    if not (0 < pa < 1):
                        raise ValueError('摩尔分数A 必须在 (0,1)')
                    if t not in TIMES:
                        raise ValueError(f'时间须为 {TIMES}')
                    pred, attn, gate = run_predict(combo, pa, comp_fp, comp_desc,
                                                   model, stats)
                    rec = {'组合': combo, '组分A': combo.split('-')[0],
                           '组分B': combo.split('-')[1], '摩尔分数A': pa,
                           '摩尔分数B': 1 - pa, '暴露时间': t,
                           f'pEC30_{t}h': round(float(pred[LABEL_COLS.index(f'{t}h_EC30')]), 4),
                           f'pEC50_{t}h': round(float(pred[LABEL_COLS.index(f'{t}h_EC50')]), 4)}
                    for j, col in enumerate(LABEL_COLS):
                        rec['预测_' + col] = round(float(pred[j]), 4)
                    rows.append(rec)
                except Exception as e:
                    errs.append(f'第 {i+2} 行：{e}')
                bar.progress((i + 1) / len(df_up))
            if errs:
                st.warning('\n'.join(errs[:10]))
            if rows:
                df_res = pd.DataFrame(rows)
                st.success(f'成功 {len(rows)} 条')
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                st.download_button('💾 下载结果 CSV',
                                   df_res.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='real_batch_predictions.csv',
                                   mime='text/csv')

st.sidebar.markdown('---')
st.sidebar.caption('真实实验数据（45 射线）训练；自定义摩尔分数为特征空间插值，'
                   '射线外组合不可预测。启动：'
                   '`python -m streamlit run web_app_real.py`')
