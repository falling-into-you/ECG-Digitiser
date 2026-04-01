"""Improved LeadIdentifier with median-based baseline removal.

PMcardio insight: using nanmedian instead of nanmean for baseline estimation
is more robust to QRS peak outliers that skew the mean.
"""

from typing import Any

import numpy as np
import torch

from src.model.lead_identifier import LeadIdentifier


class LeadIdentifierImproved(LeadIdentifier):
    """LeadIdentifier with nanmedian baseline removal.

    The only change from the parent class is in normalize():
    ``lines - lines.nanmedian(dim=1).values`` instead of ``lines - lines.nanmean(dim=1)``.
    This prevents large QRS peaks from biasing the baseline estimate.
    """

    def normalize(self, lines: torch.Tensor, avg_pixel_per_mm: float, mv_per_mm: float) -> torch.Tensor:
        """Changes the units of the ECG signals from pixels to µV, using median baseline."""
        scale = (mv_per_mm / avg_pixel_per_mm) * 1000

        # KEY CHANGE: nanmedian instead of nanmean for robust baseline
        baseline = lines.nanmedian(dim=1, keepdim=True).values
        lines = lines - baseline

        lines = lines * scale

        # Crop to columns where at least `required_valid_samples` leads are valid
        non_nan_samples_per_column = torch.sum(~torch.isnan(lines), dim=0).numpy()
        first_valid_index: int = int(np.argmax(non_nan_samples_per_column >= self.required_valid_samples))
        last_valid_index: int = int(np.argmax(non_nan_samples_per_column[::-1] >= self.required_valid_samples))
        last_valid_index = lines.shape[1] - last_valid_index - 1
        if first_valid_index <= last_valid_index:
            lines = lines[:, first_valid_index : last_valid_index + 1]

        lines = self._interpolate_lines(lines, self.target_num_samples)

        return lines
