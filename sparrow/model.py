import torch
import torch.nn as nn
import torch.nn.functional as F


class SPARROW(nn.Module):
    """Differentiable SPARROW model for stream nitrogen load routing.

    Implements the SPARROW (SPAtially Referenced Regressions On Watershed
    attributes) mass-balance routing in PyTorch, enabling gradient-based
    learning of source export and attenuation parameters.
    """

    def __init__(self):
        super(SPARROW, self).__init__()

    def forward(self, S, Z_D, Z_S, Z_R, reach_type, hydroseq, headflag,
                from_node, to_node, alpha, theta_D, theta_S, theta_R,
                frac, catchment_id, original_catchment_id, dlvdsgn, iftran):
        """
        Parameters
        ----------
        S : Tensor [N, jjsrc]
            Source strengths (e.g., total N surplus, kg/yr).
        Z_D : Tensor [N, jjdlv]
            Delivery explanatory variables (mean-centred).
        Z_S : Tensor [N, 1]
            Stream-decay explanatory variable (e.g., stream length).
        Z_R : Tensor [N, 1]
            Reservoir-decay explanatory variable (e.g., reservoir load).
        reach_type : Tensor [N]      Reach type flag.
        hydroseq : Tensor [N]
            Negative hydrologic sequence; ascending sort → upstream-to-downstream.
        headflag : Tensor [N]        1 = headwater reach.
        from_node : Tensor [N]       Upstream node ID.
        to_node : Tensor [N]         Downstream node ID.
        alpha : Tensor [N, jjsrc]    Spatially varying source export coefficients.
        theta_D : Tensor [N, jjdlv]  Delivery coefficients.
        theta_S : Tensor [N, 1]      Stream attenuation coefficient.
        theta_R : Tensor [N, 1]      Reservoir attenuation coefficient.
        frac : Tensor [N]            Fraction of reach load routed downstream.
        catchment_id : Tensor [N]    Sequential IDs (1..N) for memory-efficient indexing.
        original_catchment_id : Tensor [N]  Original waterid values.
        dlvdsgn : Tensor [jjsrc, jjdlv]    Source-delivery design matrix.
        iftran : Tensor [N]          In-transit indicator; 0 = terminal reach.

        Returns
        -------
        catchment_id, incddsrc_nd, incddsrc, total_load, original_catchment_id
        """
        # Process reaches in upstream → downstream order
        order = torch.argsort(hydroseq, descending=False)
        S = S[order]; Z_D = Z_D[order]; Z_S = Z_S[order]; Z_R = Z_R[order]
        reach_type = reach_type[order]; headflag = headflag[order]
        from_node = from_node[order]; to_node = to_node[order]
        frac = frac[order]; catchment_id = catchment_id[order]
        original_catchment_id = original_catchment_id[order]
        iftran = iftran[order]
        alpha = alpha[order]; theta_D = theta_D[order]
        theta_S = theta_S[order]; theta_R = theta_R[order]

        N = S.shape[0]

        # In-reach decay factors
        rchdcayf = torch.exp(-Z_S * theta_S)           # stream decay   [N, 1]
        resdcayf = (1 + Z_R * theta_R).pow(-1)         # reservoir decay [N, 1]
        incdecay = rchdcayf.pow(0.5) * resdcayf        # combined half-reach

        # Source × delivery factors
        ddliv1 = Z_D * theta_D                                          # [N, jjdlv]
        ddliv2 = torch.exp(torch.matmul(ddliv1, dlvdsgn.T))             # [N, jjsrc]
        dddliv = (ddliv2 * S * alpha).sum(dim=1)                        # [N]

        incddsrc_nd = dddliv
        incddsrc = (incdecay * dddliv.view(-1, 1)).squeeze()            # [N]

        # Downstream carry factor (clamp to avoid numerical zero)
        epsilon = 1e-6
        carryf = torch.clamp(
            (frac.view(-1) * rchdcayf.view(-1) * resdcayf.view(-1)).squeeze(),
            min=epsilon,
        )

        # Node-based load accumulation
        all_nodes = torch.cat([from_node, to_node])
        min_nid = all_nodes.min().item()
        nnode = int(all_nodes.max().item() - min_nid + 1)
        node_loads = torch.zeros(nnode, dtype=torch.float32)
        fn_idx = (from_node - min_nid).long()
        tn_idx = (to_node - min_nid).long()
        pred_loads = torch.zeros(N, dtype=torch.float32)

        for i in range(N):
            rchld = incddsrc[i] + carryf[i] * node_loads[fn_idx[i]]
            pred_loads[i] = rchld
            node_loads = node_loads.clone()
            node_loads[tn_idx[i]] = node_loads[tn_idx[i]] + iftran[i] * rchld

        return catchment_id, incddsrc_nd, incddsrc, pred_loads, original_catchment_id


class ParamGenerator(nn.Module):
    """3-layer MLP that generates spatially varying SPARROW parameters.

    Used for three separate sub-networks in dPL-SPARROW:

    1. **Catchment model** (`param_model`)
       Input: landscape attributes (precipitation, soil, land use, conservation).
       Outputs: N export rate (α) and delivery coefficients (θ_D).

    2. **Stream model** (`param_model_strm`)
       Input: stream-specific attributes (slope, mean discharge).
       Output (used): stream attenuation coefficient → ``coeffs[:, -2:-1]``.

    3. **Reservoir model** (`param_model_res`)
       Input: reservoir-specific attribute (mean temperature).
       Output (used): reservoir attenuation coefficient → ``coeffs[:, -1:]``.

    Parameter bounds (via output activations):
    - Source export (α): sigmoid → [0, 1]
    - Delivery (θ_D):    0.1 × tanh → [−0.1, 0.1]
    - Stream loss:       softplus → [0, ∞)
    - Reservoir loss:    softplus → [0, ∞)
    """

    def __init__(self, input_size, hidden_size=16,
                 num_source=1, num_delivery=1, num_stm_loss=1, num_res_loss=1):
        super(ParamGenerator, self).__init__()
        self.num_source = num_source
        self.num_delivery = num_delivery
        self.num_stm_loss = num_stm_loss
        self.num_res_loss = num_res_loss
        total = num_source + num_delivery + num_stm_loss + num_res_loss

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, total)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        raw = self.fc3(h)

        idx = 0
        src = torch.sigmoid(raw[:, idx:idx + self.num_source])
        idx += self.num_source

        dlv = 0.1 * torch.tanh(raw[:, idx:idx + self.num_delivery])
        idx += self.num_delivery

        stm = F.softplus(raw[:, idx:idx + self.num_stm_loss])
        idx += self.num_stm_loss

        res = F.softplus(raw[:, idx:idx + self.num_res_loss])

        return torch.cat([src, dlv, stm, res], dim=1)
