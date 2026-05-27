import random
import numpy as np
import torch
import pandas as pd


def setup_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hydseq(indata):
    """Compute hydrologic sequencing numbers (HYDSEQ) for stream network routing.

    Traverses the network from outlet to headwaters and assigns a decreasing
    integer sequence.  Sorting in ascending order processes reaches from
    upstream to downstream.  Returned values are negated to match the sign
    convention used in RSparrow.

    Parameters
    ----------
    indata : pd.DataFrame
        Must contain columns: ``waterid``, ``fnode``, ``tnode``.

    Returns
    -------
    pd.DataFrame
        Input data with added columns ``seqvar`` (0-based row index) and
        ``hydseq`` (negative integer routing order).
    """
    print("Running calculation of HYDSEQ (hydrologic sequencing numbers)...")
    data1 = indata.copy().reset_index(drop=True)
    data1['seqvar'] = range(len(data1))

    fnode_to_seqvars = data1.groupby('fnode')['seqvar'].apply(list).to_dict()
    tnode_to_seqvars = data1.groupby('tnode')['seqvar'].apply(list).to_dict()

    # Terminal reaches: their tnode does not appear as anyone's fnode
    terminal_reaches = data1[~data1['tnode'].isin(data1['fnode'])]['seqvar'].tolist()

    hydseqvar = [None] * len(data1)
    processed = set()
    counter = 0
    stack = terminal_reaches.copy()

    def upstream_ready(current_stack):
        """Return upstream reaches whose all downstream neighbours are assigned."""
        result = []
        for sv in current_stack:
            fnode = data1.loc[sv, 'fnode']
            for up in tnode_to_seqvars.get(fnode, []):
                if up not in processed:
                    tnode = data1.loc[up, 'tnode']
                    downstream = fnode_to_seqvars.get(tnode, [])
                    if all(hydseqvar[ds] is not None for ds in downstream):
                        result.append(up)
        return result

    while stack:
        h1 = counter + len(stack)
        for i, sv in enumerate(stack):
            hydseqvar[sv] = h1 - i
            processed.add(sv)
        counter = h1
        stack = upstream_ready(stack)

    data1['hydseq'] = [-x if x is not None else None for x in hydseqvar]
    return data1


def gpu_align(total_load, observed_batch, pred_ids, obs_ids):
    """Match predicted and observed loads by waterid.

    Parameters
    ----------
    total_load : Tensor [N]       Model-predicted loads for all reaches.
    observed_batch : Tensor [M]   Observed loads; NaN where ungauged.
    pred_ids : Tensor [N]         waterid for each predicted reach.
    obs_ids : Tensor [M]          waterid for each observed entry.

    Returns
    -------
    aligned_pred : Tensor   Predicted loads at matched reaches.
    aligned_obs  : Tensor   Corresponding observed loads.
    mask         : BoolTensor  True where observed value is not NaN.
    """
    matched = (obs_ids.unsqueeze(1) == pred_ids).nonzero(as_tuple=True)
    obs_idx, pred_idx = matched[0], matched[1]
    aligned_obs = observed_batch[obs_idx]
    aligned_pred = total_load[pred_idx]
    mask = ~torch.isnan(aligned_obs)
    return aligned_pred, aligned_obs, mask


# ── Metrics ────────────────────────────────────────────────────────────────

EPS = 1e-8  # small constant for numerical stability in log / R²


def r2_torch(y_true, y_pred):
    """Nash-Sutcliffe efficiency (R²) on original scale."""
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1.0 - ss_res / (ss_tot + EPS)


def r2_logspace_torch(y_true, y_pred):
    """Nash-Sutcliffe efficiency (R²) computed on log-transformed values."""
    yt = torch.log(torch.clamp(y_true, min=EPS))
    yp = torch.log(torch.clamp(y_pred, min=EPS))
    return r2_torch(yt, yp)


def weighted_means_by_group(df, group_col, value_cols, weight_col):
    """Compute area-weighted column means within groups.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str        Column to group by (e.g., ``'HUC8'``).
    value_cols : list[str] Columns to average.
    weight_col : str       Weight column (e.g., drainage area km²).

    Returns
    -------
    pd.DataFrame  Weighted group means indexed by ``group_col``.
    """
    df = df.copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
    df.loc[~np.isfinite(df[weight_col]) | (df[weight_col] < 0), weight_col] = np.nan

    def _wavg(g):
        w = g[weight_col].to_numpy()
        X = g[value_cols]
        wx_sum = X.multiply(w, axis=0).where(np.isfinite(X)).sum(axis=0, skipna=True)
        w_sum = (~X.isna()).multiply(w, axis=0).sum(axis=0, skipna=True)
        out = wx_sum / w_sum.replace(0, np.nan)
        fallback = X.mean(axis=0, skipna=True)
        return out.where(np.isfinite(out), fallback)

    return df.groupby(group_col, dropna=False).apply(_wavg)
