"""Improved SignalExtractor — iteratively optimized for ECG digitization accuracy.

Iteration 3: Endpoint extrapolation to fill small gaps at line endpoints.
Iteration 5: Internal NaN gap interpolation after merging.
Iteration 56: Coverage-weighted merge instead of first-value selection.
Viterbi DP: Optional dynamic-programming path extraction for robustness
  against overlapping leads (inspired by arxiv:2506.10617).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.model.signal_extractor import SignalExtractor


class SignalExtractorImproved(SignalExtractor):
    """SignalExtractor with endpoint extrapolation, coverage-weighted merge,
    and optional Viterbi DP path extraction."""

    def __init__(
        self,
        *,
        threshold_sum: float = 10.0,
        threshold_line_in_mask: float = 0.95,
        label_thresh: float = 0.1,
        max_iterations: int = 4,
        split_num_stripes: int = 4,
        candidate_span: int = 10,
        debug: int = 0,
        lam: float = 0.5,
        min_line_width: int = 30,
        slope_window: int = 15,
        max_extrapolation_gap: int = 30,
        max_interp_gap: int = 20,
        use_viterbi: bool = False,
        viterbi_alpha: float = 0.5,
        viterbi_prob_thresh: float = 0.1,
        viterbi_max_interp: int = 5,
    ) -> None:
        super().__init__(
            threshold_sum=threshold_sum,
            threshold_line_in_mask=threshold_line_in_mask,
            label_thresh=label_thresh,
            max_iterations=max_iterations,
            split_num_stripes=split_num_stripes,
            candidate_span=candidate_span,
            debug=debug,
            lam=lam,
            min_line_width=min_line_width,
        )
        self.slope_window = slope_window
        self.max_extrapolation_gap = max_extrapolation_gap
        self.max_interp_gap = max_interp_gap
        self.use_viterbi = use_viterbi
        self.viterbi_alpha = viterbi_alpha
        self.viterbi_prob_thresh = viterbi_prob_thresh
        self.viterbi_max_interp = viterbi_max_interp

    def _extract_line_from_region(self, fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Always use centroid extraction. Viterbi refinement is applied at merge stage."""
        return super()._extract_line_from_region(fmap, mask)

    def _viterbi_extract(self, fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi dynamic-programming path extraction.

        For each column, find contiguous signal regions in the masked probability
        map. Build a graph column-by-column and find the globally optimal path
        that minimises a cost combining distance and angle change.

        Inspired by arxiv:2506.10617.
        """
        H, W = fmap.shape
        prob = fmap * mask.float()
        thresh = self.viterbi_prob_thresh
        alpha = self.viterbi_alpha

        # --- Step 1: find candidate nodes per column ---
        # Each node = (center_y, mean_prob) of a contiguous signal region
        nodes_per_col: list[list[tuple[float, float]]] = []
        for col in range(W):
            col_vals = prob[:, col].numpy()
            nodes: list[tuple[float, float]] = []
            in_region = False
            start = 0
            for row in range(H):
                if col_vals[row] > thresh:
                    if not in_region:
                        start = row
                        in_region = True
                else:
                    if in_region:
                        seg = col_vals[start:row]
                        center = start + float(np.average(np.arange(len(seg)), weights=seg))
                        mean_p = float(seg.mean())
                        nodes.append((center, mean_p))
                        in_region = False
            if in_region:
                seg = col_vals[start:]
                center = start + float(np.average(np.arange(len(seg)), weights=seg))
                mean_p = float(seg.mean())
                nodes.append((center, mean_p))
            nodes_per_col.append(nodes)

        # --- Step 2: Viterbi DP ---
        # dp[col] = list of (cumulative_cost, prev_node_idx, prev_angle) per node
        INF = float("inf")
        line = torch.full((W,), float("nan"))

        # Find first column with nodes
        first_col = -1
        for c in range(W):
            if nodes_per_col[c]:
                first_col = c
                break
        if first_col < 0:
            return line

        # Initialize DP at first_col
        n_nodes = len(nodes_per_col[first_col])
        dp_cost = [0.0] * n_nodes
        dp_prev = [-1] * n_nodes
        dp_angle = [0.0] * n_nodes
        dp_prev_col = [first_col] * n_nodes

        prev_col_idx = first_col
        prev_dp_cost = dp_cost
        prev_dp_prev = dp_prev
        prev_dp_angle = dp_angle
        prev_dp_prev_col = dp_prev_col
        prev_nodes = nodes_per_col[first_col]

        # Store backtrack info: for each column with nodes, store
        # (nodes, best_cost, backptr_col, backptr_node_idx)
        bt_cols: list[int] = [first_col]
        bt_nodes: list[list[tuple[float, float]]] = [prev_nodes]
        bt_cost: list[list[float]] = [prev_dp_cost]
        bt_back_col: list[list[int]] = [[-1] * n_nodes]
        bt_back_node: list[list[int]] = [[-1] * n_nodes]

        for col in range(first_col + 1, W):
            cur_nodes = nodes_per_col[col]
            if not cur_nodes:
                continue

            dx = float(col - prev_col_idx)
            n_cur = len(cur_nodes)
            n_prev = len(prev_nodes)

            cur_cost = [INF] * n_cur
            cur_back_col = [-1] * n_cur
            cur_back_node = [-1] * n_cur
            cur_angle = [0.0] * n_cur

            for j in range(n_cur):
                cy = cur_nodes[j][0]
                cp = cur_nodes[j][1]  # mean probability of this node
                for k in range(n_prev):
                    py = prev_nodes[k][0]
                    dy = cy - py
                    dist = np.sqrt(dx * dx + dy * dy)
                    new_angle = np.arctan2(dy, dx)
                    angle_change = abs(new_angle - prev_dp_angle[k])
                    # Cost: distance + smoothness + inverse probability
                    # The (1/cp) term strongly favours high-probability nodes
                    inv_prob = 1.0 / max(cp, 0.01)
                    cost = (
                        prev_dp_cost[k]
                        + alpha * dist * inv_prob
                        + (1.0 - alpha) * angle_change
                    )
                    if cost < cur_cost[j]:
                        cur_cost[j] = cost
                        cur_back_col[j] = len(bt_cols) - 1  # index into bt arrays
                        cur_back_node[j] = k
                        cur_angle[j] = new_angle

            bt_cols.append(col)
            bt_nodes.append(cur_nodes)
            bt_cost.append(cur_cost)
            bt_back_col.append(cur_back_col)
            bt_back_node.append(cur_back_node)

            prev_col_idx = col
            prev_dp_cost = cur_cost
            prev_dp_angle = cur_angle
            prev_nodes = cur_nodes

        # --- Step 3: Backtrack ---
        if not bt_cost:
            return line

        # Find best endpoint
        last_costs = bt_cost[-1]
        best_node = int(np.argmin(last_costs))

        # Backtrack through bt arrays
        path: list[tuple[int, float]] = []  # (col, y)
        bt_idx = len(bt_cols) - 1
        node_idx = best_node
        while bt_idx >= 0 and node_idx >= 0:
            col = bt_cols[bt_idx]
            y = bt_nodes[bt_idx][node_idx][0]
            path.append((col, y))
            prev_bt = bt_back_col[bt_idx][node_idx]
            prev_nd = bt_back_node[bt_idx][node_idx]
            bt_idx = prev_bt
            node_idx = prev_nd

        path.reverse()

        # --- Step 4: Fill line, interpolate small gaps ---
        for col, y in path:
            line[col] = y

        # Interpolate gaps between path points (columns with no nodes)
        for i in range(len(path) - 1):
            c1, y1 = path[i]
            c2, y2 = path[i + 1]
            gap = c2 - c1 - 1
            if 0 < gap <= self.viterbi_max_interp:
                for g in range(1, gap + 1):
                    t = g / (gap + 1)
                    line[c1 + g] = y1 + t * (y2 - y1)

        return line

    def _extrapolate_endpoints(self, lines: torch.Tensor) -> torch.Tensor:
        """Linearly extrapolate short gaps at line endpoints."""
        lines = lines.clone()
        window = self.slope_window
        max_gap = self.max_extrapolation_gap

        for i in range(lines.shape[0]):
            valid = ~torch.isnan(lines[i])
            indices = valid.nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                continue
            first = int(indices[0].item())
            last = int(indices[-1].item())

            # Extrapolate left
            if first > 0 and first <= max_gap:
                left_idx = indices[:window]
                if len(left_idx) >= 2:
                    x = left_idx.float()
                    y = lines[i, left_idx]
                    xm, ym = x.mean(), y.mean()
                    var_x = ((x - xm) ** 2).sum()
                    if var_x > 1e-8:
                        slope = float(((x - xm) * (y - ym)).sum() / var_x)
                        y0 = float(lines[i, first])
                        for px in range(first - 1, -1, -1):
                            lines[i, px] = y0 + slope * (px - first)

            # Extrapolate right
            W = lines.shape[1]
            if last < W - 1 and (W - 1 - last) <= max_gap:
                right_idx = indices[-window:]
                if len(right_idx) >= 2:
                    x = right_idx.float()
                    y = lines[i, right_idx]
                    xm, ym = x.mean(), y.mean()
                    var_x = ((x - xm) ** 2).sum()
                    if var_x > 1e-8:
                        slope = float(((x - xm) * (y - ym)).sum() / var_x)
                        y0 = float(lines[i, last])
                        for px in range(last + 1, W):
                            lines[i, px] = y0 + slope * (px - last)

        return lines

    def match_and_merge_lines(self, lines: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        """Override: extrapolate endpoints before merging, with weighted merge."""
        lines = self.preprocess_lines(lines)
        lines = self._extrapolate_endpoints(lines)
        if self.debug:
            self.plot_lines(lines, "Preprocessed Lines (Improved)")
        min_coords, max_coords, heights, W = self.extract_graph_params(lines)
        cost_matrix, wrapped_mask = self.compute_cost_matrix(min_coords, max_coords, W, heights)

        row_ind, col_ind = self.match_lines(cost_matrix)

        valid_mask = ~wrapped_mask[row_ind, col_ind]
        row_ind, col_ind = row_ind[valid_mask], col_ind[valid_mask]

        graph = self.build_match_graph(row_ind, col_ind)
        components = self.get_connected_components(graph)

        merged_lines, overlaps = self._merge_components_weighted(lines, components)
        merged_lines = self._interpolate_internal_gaps(merged_lines)
        filtered_lines = [line for line in merged_lines if torch.sum(~torch.isnan(line)) >= W // 5]

        if self.debug:
            self.plot_graph(min_coords, max_coords, row_ind, col_ind)

        return filtered_lines, overlaps

    def _merge_components_weighted(
        self, lines: torch.Tensor, components: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Merge with coverage-weighted average; optionally use continuity-based
        selection in overlap columns when use_viterbi is enabled."""
        merged: list[torch.Tensor] = []
        overlaps: list[float] = []
        for group in components:
            group_lines = lines[torch.tensor(group)]
            valid_mask = ~torch.isnan(group_lines)
            merged_line = torch.full((group_lines.shape[1],), float("nan"))
            overlap = valid_mask.sum(0)
            overlaps.append(float(overlap[overlap > 0].float().mean().item()))

            coverages = valid_mask.sum(dim=1).float()
            total_coverage = coverages.sum()
            if total_coverage < 1:
                merged.append(merged_line)
                continue
            weights = coverages / total_coverage

            for col in range(group_lines.shape[1]):
                valid_values = group_lines[:, col][~torch.isnan(group_lines[:, col])]
                valid_indices = (~torch.isnan(group_lines[:, col])).nonzero(as_tuple=True)[0]
                if len(valid_values) == 0:
                    continue
                elif len(valid_values) == 1:
                    merged_line[col] = valid_values[0]
                else:
                    w = weights[valid_indices]
                    w = w / w.sum()
                    merged_line[col] = (valid_values * w).sum()

            # Viterbi-inspired refinement: in overlap regions, replace the
            # blended value with the candidate closest to the local trend
            # estimated from non-overlap neighbours.
            if self.use_viterbi and len(group) > 1:
                merged_line = self._refine_overlap_by_continuity(
                    merged_line, group_lines, valid_mask
                )

            merged.append(merged_line)
        return merged, overlaps

    def _refine_overlap_by_continuity(
        self,
        merged_line: torch.Tensor,
        group_lines: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """In columns where multiple lines overlap, replace the weighted average
        with the candidate value most consistent with the local non-overlap trend.

        Strategy: for each overlap column, estimate the expected y from nearby
        non-overlap columns (where only one line contributes) using linear
        interpolation, then pick the candidate closest to that expectation.
        """
        W = merged_line.shape[0]
        overlap_count = valid_mask.sum(0)  # (W,) how many lines valid per col
        refined = merged_line.clone()

        # Find overlap regions (count > 1)
        is_overlap = overlap_count > 1

        if not is_overlap.any():
            return refined

        # For each overlap column, find nearest non-overlap anchors
        non_overlap_cols = ((overlap_count == 1) & (~torch.isnan(merged_line))).nonzero(as_tuple=True)[0]
        if len(non_overlap_cols) < 2:
            return refined  # can't estimate trend without anchors

        non_overlap_x = non_overlap_cols.float().numpy()
        non_overlap_y = merged_line[non_overlap_cols].numpy()

        for col in range(W):
            if not is_overlap[col]:
                continue
            valid_values = group_lines[:, col][~torch.isnan(group_lines[:, col])]
            if len(valid_values) < 2:
                continue

            # Estimate expected y by interpolating from non-overlap anchors
            col_f = float(col)
            # Find nearest left and right anchors
            left_mask = non_overlap_x <= col_f
            right_mask = non_overlap_x >= col_f
            if not left_mask.any() or not right_mask.any():
                continue

            left_idx = int(np.where(left_mask)[0][-1])
            right_idx = int(np.where(right_mask)[0][0])

            if left_idx == right_idx:
                expected_y = non_overlap_y[left_idx]
            else:
                x0, y0 = non_overlap_x[left_idx], non_overlap_y[left_idx]
                x1, y1 = non_overlap_x[right_idx], non_overlap_y[right_idx]
                t = (col_f - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.5
                expected_y = y0 + t * (y1 - y0)

            # Pick candidate closest to expected
            dists = torch.abs(valid_values - expected_y)
            best = valid_values[dists.argmin()]
            refined[col] = best

        return refined

    def _interpolate_internal_gaps(self, lines: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fill small internal NaN gaps with linear interpolation."""
        result = []
        max_gap = self.max_interp_gap
        for line in lines:
            line = line.clone()
            valid = ~torch.isnan(line)
            indices = valid.nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                result.append(line)
                continue

            first = int(indices[0].item())
            last = int(indices[-1].item())
            for j in range(first, last):
                if not torch.isnan(line[j]):
                    continue
                gap_end = j
                while gap_end <= last and torch.isnan(line[gap_end]):
                    gap_end += 1
                gap_len = gap_end - j
                if gap_len <= max_gap and j > 0 and gap_end < line.shape[0]:
                    y_left = float(line[j - 1])
                    y_right = float(line[gap_end])
                    for k in range(j, gap_end):
                        t = (k - j + 1) / (gap_len + 1)
                        line[k] = y_left + t * (y_right - y_left)

            result.append(line)
        return result
