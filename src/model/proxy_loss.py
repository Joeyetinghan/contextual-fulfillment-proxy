import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class ProxyLoss(nn.Module):
    """
    Unified (structure-aware) proxy loss for (DC, Carrier) option logits.

    sel_logits : [B, D*C] - unified option logits
    qty_true   : [B, D*C] - ground-truth quantities per option
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        selection_weight: float = 1.0,
        carrier_loss_weight: float = 1.0,
        constraint_weight: float = 1.0,
        cardinality_weight: float = 0.0,
        entropy_weight: float = 0.0,
        gumbel_tau: float = 1.0,
        cost_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
        aux_dc_weight: float = 0.0,
        aux_carrier_weight: float = 0.0,
        dc_class_weights: Optional[torch.Tensor] = None,
        carrier_class_weights: Optional[torch.Tensor] = None,
        use_eligibility_mask: bool = True,
    ):
        super().__init__()
        if class_weights is None:
            cw_tensor = torch.empty(0)
        elif isinstance(class_weights, torch.Tensor):
            cw_tensor = class_weights.detach().clone().float()
        self.register_buffer("class_weights", cw_tensor)
        self.selection_weight = float(selection_weight)
        self.carrier_loss_weight = float(carrier_loss_weight)
        self.constraint_weight = float(constraint_weight)
        self.cardinality_weight = float(cardinality_weight)
        self.entropy_weight = float(entropy_weight)
        self.gumbel_tau = float(gumbel_tau)
        self.cost_loss_weight = float(cost_loss_weight)
        self.label_smoothing = float(label_smoothing)
        self.aux_dc_weight = float(aux_dc_weight)
        self.aux_carrier_weight = float(aux_carrier_weight)
        self.register_buffer("dc_class_weights", dc_class_weights if dc_class_weights is not None else torch.empty(0))
        self.register_buffer("carrier_class_weights", carrier_class_weights if carrier_class_weights is not None else torch.empty(0))
        self.use_eligibility_mask = bool(use_eligibility_mask)
        self._printed_components = False

    def forward(
        self,
        sel_logits,
        qty_true,
        inventory_vec,
        demand_scalar,
        scenario_costs=None,
        eligibility_mask=None,
    ):
        # Handle hierarchical tuple outputs: (logits_dc, logits_carrier)
        if isinstance(sel_logits, tuple):
            return self._forward_hierarchical(
                sel_logits, qty_true, inventory_vec, demand_scalar, scenario_costs, eligibility_mask
            )
        # Standard unified
        return self._forward_unified(
            sel_logits, qty_true, inventory_vec, demand_scalar, scenario_costs, eligibility_mask
        )

    def _forward_unified(self, sel_logits, qty_true, inventory_vec, demand_scalar, scenario_costs=None, eligibility_mask=None):
        """
        Unified (structure-aware) loss computation.

        Args:
            sel_logits: [B, D*C] - unified scores for all (DC, Carrier) options
            qty_true: [B, D*C] - ground truth quantities for all options
            inventory_vec: [B, D] - inventory per DC (for constraint violation loss)
            demand_scalar: [B, 1] - demand quantity (for constraint violation loss)
            scenario_costs: [B, E, D*C] - scenario costs (optional, for expected cost loss)
            eligibility_mask: [B, D*C] - binary mask for eligible options (1=eligible, 0=ineligible)

        Returns:
            total_loss: scalar loss
        """
        # Cast float16 inputs to float32 for numerical stability
        sel_logits = sel_logits.float()
        qty_true = qty_true.float()
        if inventory_vec is not None:
            inventory_vec = inventory_vec.float()
        if demand_scalar is not None:
            demand_scalar = demand_scalar.float()
        if scenario_costs is not None:
            scenario_costs = scenario_costs.float()
        if eligibility_mask is not None:
            eligibility_mask = eligibility_mask.float()

        B, DC = sel_logits.shape
        D = inventory_vec.size(1) if inventory_vec is not None else None
        C = DC // D if D is not None else None

        if not self.use_eligibility_mask:
            eligibility_mask = None

        # Mask ineligible options
        if eligibility_mask is not None:
            eligibility_mask = eligibility_mask.to(sel_logits.device)
            if eligibility_mask.shape != sel_logits.shape:
                if eligibility_mask.ndim == 1:
                    eligibility_mask = eligibility_mask.unsqueeze(0).expand(B, -1)
            # Ensure targets are always treated as eligible to avoid masking true labels.
            target_support = (qty_true > 0).float()
            if os.getenv("PROXY_LOSS_DEBUG") == "1":
                mismatch = (target_support > 0) & (eligibility_mask <= 0)
                if mismatch.any() and not self._printed_components:
                    mismatch_rows = mismatch.any(dim=1).sum().item()
                    mismatch_total = mismatch.sum().item()
                    print(f"[proxy_loss_debug] target_ineligible rows={int(mismatch_rows)} entries={int(mismatch_total)}")
            eligibility_mask = torch.maximum(eligibility_mask, target_support)
            has_eligible = eligibility_mask.sum(dim=1) > 0
            neg_inf = -1e9
            sel_logits = torch.where(eligibility_mask > 0, sel_logits, torch.full_like(sel_logits, neg_inf))
        else:
            has_eligible = torch.ones(B, dtype=torch.bool, device=sel_logits.device)

        # Drop samples with no finite eligible logits
        if eligibility_mask is not None:
            finite_mask = torch.isfinite(sel_logits) & (eligibility_mask > 0)
            has_finite = finite_mask.view(B, -1).any(dim=1)
        else:
            has_finite = torch.isfinite(sel_logits).view(B, -1).any(dim=1)

        # Selection loss
        with torch.no_grad():
            if eligibility_mask is not None:
                qty_true_masked = qty_true * eligibility_mask
                y_sum = qty_true_masked.sum(1)
                active = (y_sum > 0).float()
                qty_true_for_argmax = qty_true_masked.clone()
                qty_true_for_argmax[eligibility_mask == 0] = -1e9
                y_idx = qty_true_for_argmax.argmax(1)
                batch_indices = torch.arange(B, device=sel_logits.device)
                y_idx_elig = eligibility_mask[batch_indices, y_idx]
                needs_fallback = (y_idx_elig == 0) & has_eligible
                if needs_fallback.any():
                    first_eligible = eligibility_mask[needs_fallback].argmax(dim=1)
                    y_idx[needs_fallback] = first_eligible
            else:
                y_sum = qty_true.sum(1)
                active = (y_sum > 0).float()
                y_idx = qty_true.argmax(1)

        valid_samples = has_eligible & (active > 0) & has_finite
        if not valid_samples.any():
            return torch.tensor(0.0, device=sel_logits.device, requires_grad=True)

        # Sanitize logits after masking
        sel_logits = torch.nan_to_num(sel_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

        weight = None
        if self.class_weights.numel() > 0:
            weight_tensor = self.class_weights.to(sel_logits.device)
            if weight_tensor.size(0) == DC:
                weight = weight_tensor
            elif D is not None and weight_tensor.size(0) == D:
                weight = weight_tensor.repeat_interleave(C)

        label_smoothing_val = self.label_smoothing if eligibility_mask is None else 0.0
        ce_elem = F.cross_entropy(
            sel_logits, y_idx,
            reduction='none',
            weight=weight,
            label_smoothing=label_smoothing_val,
        )
        ce_elem = ce_elem * valid_samples.float()
        active_valid = active * valid_samples.float()
        sel_loss = ce_elem.sum() / active_valid.sum().clamp_min(1.0)

        # Auxiliary DC/Carrier loss (conditioned on true DC)
        aux_loss = 0.0
        aux_dc_loss = 0.0
        aux_carrier_loss = 0.0
        if (self.aux_dc_weight > 0 or self.aux_carrier_weight > 0) and D is not None:
            logits_grid = sel_logits.view(B, D, C)
            qty_grid = qty_true.view(B, D, C)
            elig_grid = None
            if eligibility_mask is not None:
                elig_grid = eligibility_mask.view(B, D, C)
                qty_grid = qty_grid * elig_grid

            qty_per_dc = qty_grid.sum(dim=2)
            if elig_grid is not None:
                elig_dc = (elig_grid.sum(dim=2) > 0)
                qty_per_dc_for_argmax = qty_per_dc.clone()
                qty_per_dc_for_argmax[~elig_dc] = -1e9
                true_dc = qty_per_dc_for_argmax.argmax(dim=1)
                batch_indices = torch.arange(B, device=sel_logits.device)
                y_dc_elig = elig_dc[batch_indices, true_dc]
                needs_fallback = (~y_dc_elig) & has_eligible
                if needs_fallback.any():
                    first_eligible = elig_dc[needs_fallback].argmax(dim=1)
                    true_dc[needs_fallback] = first_eligible
            else:
                true_dc = qty_per_dc.argmax(dim=1)

            if self.aux_dc_weight > 0:
                dc_logits = torch.logsumexp(logits_grid, dim=2)
                if eligibility_mask is not None:
                    neg_inf = -1e9
                    dc_logits = torch.where(elig_dc, dc_logits, torch.full_like(dc_logits, neg_inf))
                dc_weight = None
                if self.dc_class_weights.numel() > 0:
                    dc_weight = self.dc_class_weights.to(dc_logits.device)
                dc_ce = F.cross_entropy(
                    dc_logits, true_dc, reduction='none', weight=dc_weight, label_smoothing=label_smoothing_val
                )
                dc_ce = dc_ce * valid_samples.float()
                aux_dc_loss = dc_ce.sum() / active_valid.sum().clamp_min(1.0)

            if self.aux_carrier_weight > 0:
                batch_indices = torch.arange(B, device=sel_logits.device)
                carrier_logits = logits_grid[batch_indices, true_dc, :]
                qty_for_true = qty_grid[batch_indices, true_dc, :]
                if elig_grid is not None:
                    elig_for_true = elig_grid[batch_indices, true_dc, :]
                    neg_inf = -1e9
                    carrier_logits = torch.where(elig_for_true > 0, carrier_logits, torch.full_like(carrier_logits, neg_inf))
                    qty_for_true = qty_for_true * elig_for_true
                    y_carrier = qty_for_true.argmax(dim=1)
                    y_carrier_elig = elig_for_true[batch_indices, y_carrier]
                    needs_fallback = (y_carrier_elig == 0) & has_eligible
                    if needs_fallback.any():
                        first_eligible = elig_for_true[needs_fallback].argmax(dim=1)
                        y_carrier[needs_fallback] = first_eligible
                else:
                    y_carrier = qty_for_true.argmax(dim=1)

                carrier_weight = None
                if self.carrier_class_weights.numel() > 0:
                    carrier_weight = self.carrier_class_weights.to(carrier_logits.device)
                carrier_ce = F.cross_entropy(
                    carrier_logits, y_carrier, reduction='none', weight=carrier_weight, label_smoothing=label_smoothing_val
                )
                carrier_ce = carrier_ce * valid_samples.float()
                aux_carrier_loss = carrier_ce.sum() / active_valid.sum().clamp_min(1.0)

            aux_loss = self.aux_dc_weight * aux_dc_loss + self.aux_carrier_weight * aux_carrier_loss

        # Constraint violation loss
        const_loss = 0.0
        if self.constraint_weight > 0 and inventory_vec is not None and demand_scalar is not None and D is not None:
            sel_logits_grid = sel_logits.view(B, D, C)
            sel_logits_dc = torch.logsumexp(sel_logits_grid, dim=2)

            elig_dc = None
            if eligibility_mask is not None:
                elig_grid = eligibility_mask.view(B, D, C)
                elig_dc = (elig_grid.sum(dim=2) > 0).float()
                neg_inf = -1e9
                sel_logits_dc = torch.where(elig_dc > 0, sel_logits_dc, torch.full_like(sel_logits_dc, neg_inf))

            if valid_samples.any():
                if self.training:
                    g_dc = F.gumbel_softmax(sel_logits_dc, tau=self.gumbel_tau, hard=True, dim=1)
                    if torch.isnan(g_dc).any():
                        g_dc = torch.zeros_like(sel_logits_dc)
                        if elig_dc is not None:
                            elig_dc_normalized = elig_dc / (elig_dc.sum(dim=1, keepdim=True).clamp_min(1.0))
                            g_dc = elig_dc_normalized
                        else:
                            g_dc[:, 0] = 1.0
                    alloc = g_dc * demand_scalar
                else:
                    if elig_dc is not None:
                        dc_has_eligible = elig_dc.sum(dim=1) > 0
                        top1_dc_idx = sel_logits_dc.argmax(dim=1)
                        if not dc_has_eligible.all():
                            first_eligible_dc = elig_dc.argmax(dim=1)
                            top1_dc_idx[~dc_has_eligible] = first_eligible_dc[~dc_has_eligible]
                    else:
                        top1_dc_idx = sel_logits_dc.argmax(dim=1)
                    one_hot_dc = torch.zeros_like(sel_logits_dc)
                    one_hot_dc.scatter_(1, top1_dc_idx.unsqueeze(1), 1.0)
                    alloc = one_hot_dc * demand_scalar

                if elig_dc is not None:
                    alloc = alloc * elig_dc

                alloc_valid = alloc * valid_samples.float().unsqueeze(1)
                demand_valid = demand_scalar * valid_samples.float()
                inventory_valid = inventory_vec * valid_samples.float().unsqueeze(1)
                fulfil = torch.minimum(alloc_valid, inventory_valid).sum(dim=1)
                const_loss = (demand_valid.squeeze(1) - fulfil).clamp_min(0)
                const_loss = const_loss.sum() / valid_samples.sum().clamp_min(1.0)
                const_loss = self.constraint_weight * const_loss
            else:
                const_loss = torch.tensor(0.0, device=sel_logits.device)

        # Cardinality penalty
        card_loss = 0.0
        if self.cardinality_weight:
            card_loss = self.cardinality_weight * (torch.sigmoid(sel_logits).sum(1) * valid_samples.float()).sum() / valid_samples.sum().clamp_min(1.0)

        # Entropy regularization
        entropy_loss = 0.0
        if self.entropy_weight and valid_samples.any():
            probs = F.softmax(sel_logits, dim=1)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            entropy_loss = -self.entropy_weight * (entropy * valid_samples.float()).sum() / valid_samples.sum().clamp_min(1.0)

        # Expected cost loss
        cost_loss_val = 0.0
        if self.cost_loss_weight > 0 and scenario_costs is not None and valid_samples.any():
            scenario_costs = torch.nan_to_num(scenario_costs, nan=0.0, posinf=0.0, neginf=0.0)
            mean_costs = scenario_costs.mean(dim=1)
            probs = F.softmax(sel_logits, dim=1)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
            expected_cost = (probs * mean_costs).sum(dim=1)
            cost_loss_val = self.cost_loss_weight * (expected_cost * valid_samples.float()).sum() / valid_samples.sum().clamp_min(1.0)

        total = self.selection_weight * sel_loss + const_loss + card_loss + entropy_loss + cost_loss_val + aux_loss
        if os.getenv("PROXY_LOSS_COMPONENTS") == "1" and not self._printed_components:
            self._printed_components = True
            def _to_float(x):
                if torch.is_tensor(x):
                    return float(x.detach().cpu())
                return float(x)
            print(
                "[loss-components] "
                f"sel={_to_float(sel_loss):.4f} "
                f"const={_to_float(const_loss):.4f} "
                f"card={_to_float(card_loss):.4f} "
                f"entropy={_to_float(entropy_loss):.4f} "
                f"cost={_to_float(cost_loss_val):.4f} "
                f"aux={_to_float(aux_loss):.4f} "
                f"valid={int(valid_samples.sum().item())}"
            )
        if os.getenv("PROXY_LOSS_DEBUG") == "1":
            if torch.isnan(total) or torch.isinf(total):
                stats = {
                    "sel_loss": float(sel_loss.detach().cpu()),
                    "const_loss": float(const_loss.detach().cpu()) if torch.is_tensor(const_loss) else float(const_loss),
                    "card_loss": float(card_loss.detach().cpu()) if torch.is_tensor(card_loss) else float(card_loss),
                    "entropy_loss": float(entropy_loss.detach().cpu()) if torch.is_tensor(entropy_loss) else float(entropy_loss),
                    "cost_loss": float(cost_loss_val.detach().cpu()) if torch.is_tensor(cost_loss_val) else float(cost_loss_val),
                    "logits_min": float(sel_logits.min().detach().cpu()),
                    "logits_max": float(sel_logits.max().detach().cpu()),
                    "valid_samples": int(valid_samples.sum().item()),
                }
                raise ValueError(f"[proxy_loss_debug] NaN/Inf total loss. stats={stats}")
        return total

    def _forward_hierarchical(self, sel_logits_tuple, qty_true, inventory_vec, demand_scalar, scenario_costs=None, eligibility_mask=None):
        """
        Hierarchical (DC + Carrier) loss computation with explicit supervision.
        
        Args:
            sel_logits_tuple: (logits_dc [B, D], logits_carrier [B, D, C])
            qty_true: [B, D*C] - ground truth flattened (DC, Carrier) grid
            inventory_vec: [B, D] - inventory per DC
            demand_scalar: [B, 1] - demand quantity
            scenario_costs: [B, S, D*C] - scenario costs (optional)
            eligibility_mask: [B, D*C] - binary mask for eligible options
        
        Returns:
            total_loss: scalar loss
        """
        logits_dc, logits_carrier = sel_logits_tuple
        
        # Cast float16 to float32
        logits_dc = logits_dc.float()
        logits_carrier = logits_carrier.float()
        qty_true = qty_true.float()
        if inventory_vec is not None:
            inventory_vec = inventory_vec.float()
        if demand_scalar is not None:
            demand_scalar = demand_scalar.float()
        if scenario_costs is not None:
            scenario_costs = scenario_costs.float()
        if eligibility_mask is not None:
            eligibility_mask = eligibility_mask.float()
        
        B, D = logits_dc.shape
        C = logits_carrier.shape[2]
        
        # Reshape qty_true from [B, D*C] to [B, D, C]
        qty_grid = qty_true.view(B, D, C)
        
        # Parse eligibility mask
        elig_grid = None
        if eligibility_mask is not None and self.use_eligibility_mask:
            eligibility_mask = eligibility_mask.to(logits_dc.device)
            elig_grid = eligibility_mask.view(B, D, C)
            elig_dc = (elig_grid.sum(dim=2) > 0).float()
        else:
            elig_dc = torch.ones(B, D, device=logits_dc.device)
        
        # ===== DC Selection Loss =====
        with torch.no_grad():
            if elig_grid is not None:
                qty_grid_masked = qty_grid * elig_grid
                qty_per_dc = qty_grid_masked.sum(dim=2)
            else:
                qty_per_dc = qty_grid.sum(dim=2)
            
            y_sum = qty_per_dc.sum(1)
            active = (y_sum > 0).float()
            
            if elig_grid is not None:
                qty_per_dc_argmax = qty_per_dc.clone()
                qty_per_dc_argmax[elig_dc == 0] = -1e9
                y_dc_idx = qty_per_dc_argmax.argmax(1)
                batch_indices = torch.arange(B, device=logits_dc.device)
                y_dc_elig = elig_dc[batch_indices, y_dc_idx]
                needs_fallback = (y_dc_elig == 0)
                if needs_fallback.any():
                    first_eligible = elig_dc[needs_fallback].argmax(dim=1)
                    y_dc_idx[needs_fallback] = first_eligible
            else:
                y_dc_idx = qty_per_dc.argmax(1)
        
        valid_samples = (active > 0)
        if not valid_samples.any():
            return torch.tensor(0.0, device=logits_dc.device, requires_grad=True)
        
        # Mask ineligible DCs
        if elig_grid is not None:
            neg_inf = -1e9
            logits_dc = torch.where(elig_dc > 0, logits_dc, torch.full_like(logits_dc, neg_inf))
        
        # DC cross-entropy
        dc_weight = None
        if self.dc_class_weights.numel() > 0:
            dc_weight = self.dc_class_weights.to(logits_dc.device)
        
        label_smoothing_val = self.label_smoothing if elig_grid is None else 0.0
        ce_dc = F.cross_entropy(
            logits_dc, y_dc_idx,
            reduction='none',
            weight=dc_weight,
            label_smoothing=label_smoothing_val
        )
        ce_dc = ce_dc * valid_samples.float()
        dc_loss = ce_dc.sum() / valid_samples.sum().clamp_min(1.0)
        
        # ===== Carrier Selection Loss (Conditioned on TRUE DC) =====
        with torch.no_grad():
            batch_indices = torch.arange(B, device=qty_grid.device)
            qty_for_true_dc = qty_grid[batch_indices, y_dc_idx, :]  # [B, C]
            
            if elig_grid is not None:
                elig_for_true_dc = elig_grid[batch_indices, y_dc_idx, :]  # [B, C]
                qty_for_true_dc = qty_for_true_dc * elig_for_true_dc
                qty_argmax = qty_for_true_dc.clone()
                qty_argmax[elig_for_true_dc == 0] = -1e9
                y_carrier_idx = qty_argmax.argmax(1)
                y_carrier_elig = elig_for_true_dc[batch_indices, y_carrier_idx]
                needs_fallback = (y_carrier_elig == 0)
                if needs_fallback.any():
                    first_eligible = elig_for_true_dc[needs_fallback].argmax(dim=1)
                    y_carrier_idx[needs_fallback] = first_eligible
            else:
                y_carrier_idx = qty_for_true_dc.argmax(1)
        
        # Extract carrier logits for true DC [B, C]
        carrier_logits_true = logits_carrier[batch_indices, y_dc_idx, :]
        
        # Mask ineligible carriers
        if elig_grid is not None:
            elig_for_true_dc = elig_grid[batch_indices, y_dc_idx, :]
            neg_inf = -1e9
            carrier_logits_true = torch.where(
                elig_for_true_dc > 0,
                carrier_logits_true,
                torch.full_like(carrier_logits_true, neg_inf)
            )
        
        carrier_weight = None
        if self.carrier_class_weights.numel() > 0:
            carrier_weight = self.carrier_class_weights.to(carrier_logits_true.device)
        ce_carrier = F.cross_entropy(
            carrier_logits_true, y_carrier_idx,
            reduction='none',
            weight=carrier_weight,
            label_smoothing=label_smoothing_val
        )
        ce_carrier = ce_carrier * valid_samples.float()
        carrier_loss = ce_carrier.sum() / valid_samples.sum().clamp_min(1.0)

        if os.getenv("PROXY_LOSS_DEBUG") == "1":
            if not torch.isfinite(carrier_loss) or carrier_loss > 1e3:
                def _stat(t):
                    return {
                        "min": float(t.min().detach().cpu()),
                        "max": float(t.max().detach().cpu()),
                        "mean": float(t.mean().detach().cpu()),
                        "finite": bool(torch.isfinite(t).all().item()),
                    }
                dbg = {
                    "carrier_loss": float(carrier_loss.detach().cpu()) if torch.is_tensor(carrier_loss) else float(carrier_loss),
                    "carrier_logits_true": _stat(carrier_logits_true),
                    "logits_carrier": _stat(logits_carrier),
                    "y_carrier_idx_min": int(y_carrier_idx.min().item()),
                    "y_carrier_idx_max": int(y_carrier_idx.max().item()),
                    "valid_samples": int(valid_samples.sum().item()),
                }
                if elig_grid is not None:
                    elig_for_true_dc = elig_grid[batch_indices, y_dc_idx, :]
                    dbg["elig_for_true_dc_min"] = float(elig_for_true_dc.min().item())
                    dbg["elig_for_true_dc_max"] = float(elig_for_true_dc.max().item())
                    dbg["elig_for_true_dc_sum_min"] = float(elig_for_true_dc.sum(dim=1).min().item())
                    dbg["elig_for_true_dc_sum_max"] = float(elig_for_true_dc.sum(dim=1).max().item())
                print(f"[proxy_loss_debug] carrier_loss_spike {dbg}")
        
        # Combined selection loss
        sel_loss = dc_loss + self.carrier_loss_weight * carrier_loss
        
        # ===== Constraint Loss =====
        const_loss = 0.0
        if self.constraint_weight > 0 and inventory_vec is not None and demand_scalar is not None:
            if self.training:
                g_dc = F.gumbel_softmax(logits_dc, tau=self.gumbel_tau, hard=True, dim=1)
                if torch.isnan(g_dc).any():
                    g_dc = torch.zeros_like(logits_dc)
                    if elig_grid is not None:
                        elig_dc_normalized = elig_dc / (elig_dc.sum(dim=1, keepdim=True).clamp_min(1.0))
                        g_dc = elig_dc_normalized
                    else:
                        g_dc[:, 0] = 1.0
                alloc = g_dc * demand_scalar
            else:
                top1_dc_idx = logits_dc.argmax(dim=1)
                one_hot_dc = torch.zeros_like(logits_dc)
                one_hot_dc.scatter_(1, top1_dc_idx.unsqueeze(1), 1.0)
                alloc = one_hot_dc * demand_scalar
            
            fulfil = torch.minimum(alloc, inventory_vec).sum(dim=1)
            const_loss = (demand_scalar.squeeze(1) - fulfil).clamp_min(0).mean()
            const_loss = self.constraint_weight * const_loss
        
        # ===== Cardinality Penalty =====
        card_loss = 0.0
        if self.cardinality_weight:
            card_loss_dc = torch.sigmoid(logits_dc).sum(1).mean()
            card_loss_carrier = torch.sigmoid(logits_carrier).sum(2).mean(1).mean()
            card_loss = self.cardinality_weight * (card_loss_dc + card_loss_carrier)
        
        # ===== Entropy Regularization =====
        entropy_loss = 0.0
        if self.entropy_weight > 0:
            probs_dc = F.softmax(logits_dc, dim=1)
            probs_carrier = F.softmax(logits_carrier, dim=2)
            entropy_dc = -(probs_dc * torch.log(probs_dc + 1e-10)).sum(dim=1).mean()
            entropy_carrier = -(probs_carrier * torch.log(probs_carrier + 1e-10)).sum(dim=2).mean(dim=1).mean()
            entropy_loss = -self.entropy_weight * (entropy_dc + entropy_carrier)
        
        # ===== Expected Cost Loss =====
        cost_loss_val = 0.0
        if self.cost_loss_weight > 0 and scenario_costs is not None:
            # scenario_costs: [B, S, D*C] -> [B, S, D, C]
            scenario_costs_grid = scenario_costs.view(B, scenario_costs.size(1), D, C)
            mean_costs = scenario_costs_grid.mean(dim=1)  # [B, D, C]
            
            # Joint probabilities: P(d,c) = P(d) * P(c|d)
            probs_dc = F.softmax(logits_dc, dim=1)  # [B, D]
            carrier_logits_for_cost = logits_carrier
            if elig_grid is not None:
                neg_inf = -1e9
                carrier_logits_for_cost = torch.where(
                    elig_grid > 0,
                    carrier_logits_for_cost,
                    torch.full_like(carrier_logits_for_cost, neg_inf)
                )
            probs_carrier = F.softmax(carrier_logits_for_cost, dim=2)  # [B, D, C]
            probs_carrier = torch.nan_to_num(probs_carrier, nan=0.0)
            probs_joint = probs_dc.unsqueeze(2) * probs_carrier  # [B, D, C]
            
            expected_cost = (probs_joint * mean_costs).sum(dim=(1, 2)).mean()
            cost_loss_val = self.cost_loss_weight * expected_cost
        
        total = self.selection_weight * sel_loss + const_loss + card_loss + entropy_loss + cost_loss_val
        
        if os.getenv("PROXY_LOSS_COMPONENTS") == "1" and not self._printed_components:
            self._printed_components = True
            print(
                f"[loss-components-hierarchical] "
                f"dc={float(dc_loss.detach().cpu()):.4f} "
                f"carrier={float(carrier_loss.detach().cpu()):.4f} "
                f"const={float(const_loss) if torch.is_tensor(const_loss) else const_loss:.4f} "
                f"card={float(card_loss) if torch.is_tensor(card_loss) else card_loss:.4f} "
                f"entropy={float(entropy_loss) if torch.is_tensor(entropy_loss) else entropy_loss:.4f} "
                f"cost={float(cost_loss_val) if torch.is_tensor(cost_loss_val) else cost_loss_val:.4f} "
                f"valid={int(valid_samples.sum().item())}"
            )
        
        return total
