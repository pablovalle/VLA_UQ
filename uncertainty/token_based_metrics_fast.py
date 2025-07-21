import numpy as np
import torch
from torch.nn.functional import softmax


class TokenMetricsFast:
    def __init__(self):

        self.shannon_entropy_list = []
        self.token_prob = []
        self.pcs = []
        self.token_prob_inv = []
        self.pcs_inv = []
        self.deepgini = []

    def calculate_metrics(self, logits):
        self.clear()

        probs = softmax(logits, dim=1).detach().to(torch.float32).cpu().numpy()

        eps = 1e-15
        log_base = np.log(2)

        # Shannon Entropy
        entropy = -np.sum(probs * np.log(probs + eps), axis=1) / log_base
        self.shannon_entropy_list = [float(f"{v:.5f}") for v in entropy]

        # Max token probability
        max_probs = np.max(probs, axis=1)
        self.token_prob = [float(f"{v:.5f}") for v in max_probs]

        # PCS (max - second max)
        sorted_probs = -np.sort(-probs, axis=1)
        pcs = sorted_probs[:, 0] - sorted_probs[:, 1]
        self.pcs = [float(f"{v:.5f}") for v in pcs]

        # DeepGini
        deepgini = 1 - np.sum(probs ** 2, axis=1)
        self.deepgini = [float(f"{v:.5f}") for v in deepgini]

        return [self.shannon_entropy_list, self.token_prob, self.pcs, self.deepgini]

    def compute_norm_inv_token_metrics(self, logits):
        """
        Compute various token-level uncertainty and confidence metrics from model logits,
        normalize them to [0, 1], and invert selected metrics so that higher values
        consistently indicate greater uncertainty.

        Metrics computed:
        - Shannon Entropy (normalized): uncertainty measure normalized by log2(num_classes).
        - Max Token Probability (normalized and inverted): confidence of top predicted token,
          normalized and inverted so higher means less confidence.
        - PCS (Prediction Confidence Score) (inverted): difference between top two token probabilities,
          inverted so higher means more uncertainty.
        - DeepGini (normalized): uncertainty measure normalized by its max possible value.

        Args:
            logits (torch.Tensor): raw output logits from the model with shape (batch_size, num_classes).

        Returns:
            list: four lists of float values rounded to 5 decimals, corresponding to:
                  [shannon_entropy, max_token_prob_inverted, pcs_inverted, deepgini]
        """
        self.clear()

        probs = softmax(logits, dim=1).detach().to(torch.float32).cpu().numpy()

        eps = 1e-15
        log_base = np.log(2)
        num_classes = probs.shape[1]

        # ---------- Shannon Entropy ----------
        # Raw range: [0, log2(C)]
        # Normalized range: [0, 1] → 0: certain, 1: most uncertain (uniform distribution)
        entropy = -np.sum(probs * np.log(probs + eps), axis=1) / log_base
        entropy_norm = entropy / np.log2(num_classes)
        self.shannon_entropy_list = [float(f"{v:.5f}") for v in entropy_norm]

        # ---------- Max Token Probability ----------
        # Raw range: [1/C, 1] → 1: confident, 1/C: uncertain
        # Normalized range: [0, 1] → 0: confident, 1: uncertain (after inversion)
        max_probs = np.max(probs, axis=1)
        max_probs_norm = (max_probs - 1.0 / num_classes) / (1 - 1.0 / num_classes)
        max_probs_inv = 1.0 - max_probs_norm
        self.token_prob_inv = [float(f"{v:.5f}") for v in max_probs_inv]

        # ---------- PCS (Prediction Confidence Score) ----------
        # Raw range: [0, 1] → 1: confident, 0: ambiguous (top-2 equal)
        # Inverted range: [0, 1] → 0: confident, 1: ambiguous
        sorted_probs = -np.sort(-probs, axis=1)
        pcs = sorted_probs[:, 0] - sorted_probs[:, 1]
        pcs_inv = 1.0 - pcs
        self.pcs_inv = [float(f"{v:.5f}") for v in pcs_inv]

        # ---------- DeepGini ----------
        # Raw range: [0, 1 - 1/C] → 0: confident, max: uniform
        # Normalized range: [0, 1] → 0: confident, 1: most uncertain
        deepgini = 1 - np.sum(probs ** 2, axis=1)
        deepgini_norm = deepgini / (1 - 1.0 / num_classes)
        self.deepgini = [float(f"{v:.5f}") for v in deepgini_norm]

        return [self.shannon_entropy_list, self.token_prob_inv, self.pcs_inv, self.deepgini]

    def clear(self):
        self.shannon_entropy_list = []
        self.token_prob = []
        self.pcs = []
        self.deepgini = []
