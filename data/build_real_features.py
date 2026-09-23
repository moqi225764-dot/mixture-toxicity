#!/usr/bin/env python
# encoding: utf-8

"""真实分子特征工程（论文 5 种抗生素 × 45 条混合射线）
输入：
  data/real_components.csv     Table S1
  data/real_mixture_rays.csv   Table S2（摩尔分数）
  data/real_mixtures_wide.csv  10 个 pEC 标签
输出：
  data/real_component_features.csv  每组分 Morgan 指纹 + RDKit 2D 描述符
  data/real_dataset.npz             45 射线的三模态特征矩阵 + 标签

说明：
- SMILES 来自 PubChem（按 CAS 查询的 ConnectivitySMILES，CID 记录在案）；
  CTC 为盐酸盐，按惯例取最大有机片段（与 Dragon 7.0 对游离碱计算一致）。
- Dragon 7.0 的 CATS2D/GATS7s/DISPe 为专有描述符无法复现，这里用 RDKit
  可公开复现的 2D 描述符替代，再按论文 eq4/eq5 用摩尔分数聚合成混合物描述符。
- 分子指纹：Morgan(radius=2, 1024 bit, count)，逐组分保留，
  模型内做对称聚合；摩尔分数在混合物描述符与暴露上下文中体现。
"""

import os
import json
import urllib.request
import ssl
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, EState

HERE = os.path.dirname(__file__)

# PubChem ConnectivitySMILES（2026-09 按 CAS 查询，CID 已核验）
SMILES = {
    'ENR': ('CCN1CCN(CC1)C2=C(C=C3C(=C2)N(C=C(C3=O)C(=O)O)C4CC4)F', 71188),
    'CTC': ('CC1(C2CC3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C(C=CC(=C41)Cl)O)O)O)O)'
            'C(=O)N)N(C)C)O.Cl', 54682468),
    'TMP': ('COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N', 5578),
    'CMP': ('C1=CC(=CC=C1C(C(CO)NC(=O)C(Cl)Cl)O)[N+](=O)[O-]', 5959),
    'ETM': ('CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)'
            'OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O', 12560),
}

FP_BITS = 1024
FP_RADIUS = 2

# 18 个公开可复现的 RDKit 2D 描述符（覆盖疏水性/极性/形状/电子/拓扑，
# 对应论文 Dragon 描述符的同类信息维度）
DESC_FUNCS = {
    'MolWt': Descriptors.MolWt,
    'MolLogP': Descriptors.MolLogP,
    'TPSA': Descriptors.TPSA,
    'MolMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
    'NumHDonors': Descriptors.NumHDonors,
    'NumHAcceptors': Descriptors.NumHAcceptors,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumAromaticRings': Descriptors.NumAromaticRings,
    'FractionCSP3': Descriptors.FractionCSP3,
    'BalabanJ': GraphDescriptors.BalabanJ,
    'BertzCT': GraphDescriptors.BertzCT,
    'HallKierAlpha': GraphDescriptors.HallKierAlpha,
    'Kappa2': GraphDescriptors.Kappa2,
    'Kappa3': GraphDescriptors.Kappa3,
    'Chi1v': GraphDescriptors.Chi1v,
    'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex,
    'MaxPartialCharge': Descriptors.MaxPartialCharge,
}
DESC_NAMES = list(DESC_FUNCS.keys())


def largest_fragment(smi):
    """盐型取最大有机片段（CTC 盐酸盐 → 金霉素游离碱）。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError('invalid SMILES: ' + smi)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def component_features(abbr, smi):
    mol = largest_fragment(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, FP_RADIUS, nBits=FP_BITS, useFeatures=False
    )
    fp_arr = np.array(fp, dtype=np.float32)
    desc = np.array([DESC_FUNCS[n](mol) for n in DESC_NAMES], dtype=np.float64)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return fp_arr, desc.astype(np.float32), Chem.MolToSmiles(mol)


def eq4_linear(p, x):
    """论文 eq4: D_mix = Σ p_i x_i。"""
    return (p[:, None] * x).sum(axis=0)


def eq5_rms(p, x):
    """论文 eq5: D_mix = (Σ p_i x_i^2)^{1/2}。"""
    return np.sqrt((p[:, None] * x ** 2).sum(axis=0))


def build_context(p_a):
    """暴露/配方上下文（替代演示版的浓度分支）。
    p_b = 1 - p_a；含线性、平方、交互与对数摩尔分数。"""
    p_b = 1.0 - p_a
    eps = 1e-6
    return np.array([
        p_a, p_b, p_a * p_b, p_a ** 2, p_b ** 2,
        -np.log10(max(p_a, eps)), -np.log10(max(p_b, eps))
    ], dtype=np.float32)


def main():
    df_comp = pd.read_csv(os.path.join(HERE, 'real_components.csv'))
    df_rays = pd.read_csv(os.path.join(HERE, 'real_mixture_rays.csv'))
    df_wide = pd.read_csv(os.path.join(HERE, 'real_mixtures_wide.csv'))
    label_cols = [f'{t}h_{e}' for t in [4, 6, 8, 10, 12]
                  for e in ['EC30', 'EC50']]

    # 1) 组分级特征
    abbrs = df_comp['abbreviation'].tolist()
    comp_fp, comp_desc, cleaned_smiles = {}, {}, {}
    feat_rows = []
    for abbr in abbrs:
        smi, cid = SMILES[abbr]
        fp, desc, smi_clean = component_features(abbr, smi)
        comp_fp[abbr] = fp
        comp_desc[abbr] = desc
        cleaned_smiles[abbr] = smi_clean
        row = {'abbreviation': abbr, 'pubchem_cid': cid,
               'smiles_used': smi_clean}
        row.update({n: float(v) for n, v in zip(DESC_NAMES, desc)})
        feat_rows.append(row)
    pd.DataFrame(feat_rows).to_csv(
        os.path.join(HERE, 'real_component_features.csv'),
        index=False, encoding='utf-8-sig'
    )

    # 2) 45 射线三模态特征
    n = len(df_rays)
    fingers = np.zeros((n, 2, FP_BITS), dtype=np.float32)   # 逐组分指纹
    mixdesc = np.zeros((n, 2 * len(DESC_NAMES)), dtype=np.float32)
    context = np.zeros((n, 7), dtype=np.float32)
    labels = df_wide.sort_values('no')[label_cols].values.astype(np.float32)

    for i, r in df_rays.sort_values('no').reset_index(drop=True).iterrows():
        a, b = r['comp_a'], r['comp_b']
        p = np.array([r['mole_frac_a'], r['mole_frac_b']], dtype=np.float64)
        x = np.stack([comp_desc[a], comp_desc[b]], axis=0)
        fingers[i, 0] = comp_fp[a]
        fingers[i, 1] = comp_fp[b]
        mixdesc[i, :len(DESC_NAMES)] = eq4_linear(p, x)
        mixdesc[i, len(DESC_NAMES):] = eq5_rms(p, x)
        context[i] = build_context(float(r['mole_frac_a']))

    np.savez(
        os.path.join(HERE, 'real_dataset.npz'),
        fingers=fingers, mixdesc=mixdesc, context=context, labels=labels,
        label_cols=np.array(label_cols),
        rays=df_rays.sort_values('no')[['no', 'combo', 'ray', 'comp_a',
                                       'comp_b', 'mole_frac_a', 'mole_frac_b']].values,
        desc_names=np.array(DESC_NAMES)
    )

    print('=== 真实特征构建完成 ===')
    print('组分:', abbrs)
    print('Morgan 指纹: %d bit/radius=%d' % (FP_BITS, FP_RADIUS))
    print('2D 描述符: %d 个 → eq4+eq5 聚合后混合物描述符 %d 维'
          % (len(DESC_NAMES), 2 * len(DESC_NAMES)))
    print('fingers:', fingers.shape, '| mixdesc:', mixdesc.shape,
          '| context:', context.shape, '| labels:', labels.shape)
    print('标签范围 pEC: %.3f ~ %.3f' % (labels.min(), labels.max()))
    print()
    print('描述符清单:', DESC_NAMES)
    print()
    print('产物: real_component_features.csv, real_dataset.npz')


if __name__ == '__main__':
    main()
