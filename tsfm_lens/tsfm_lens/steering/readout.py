"""Construct rank-limited input edits from the rows used by a forecast readout."""
import numpy as np


class ReadoutSpace:
    def __init__(self, rows):
        u, s, vt = np.linalg.svd(rows, full_matrices=False)
        keep = s > s[0] * max(rows.shape) * np.finfo(float).eps
        self.axes = vt[keep]
        self.inverse = u[:, keep].T / s[keep, None]
        self.forward = u[:, keep] * s[keep]

    def basis(self, targets, rank):
        """Prioritize minimum-norm edits for calibration targets, then their stepwise components.

        Separating steps lets an unseen time shape use the same channel pattern.
        Operate in compact singular-vector coordinates and keep the configured rank cap.
        """
        rank = min(rank, len(self.axes))
        if rank == len(self.axes):
            return self.axes, self.forward
        targets = np.asarray(targets)
        n, h, q = targets.shape
        edits = self.inverse @ targets.reshape(n, -1).T
        per_step = np.einsum("rhq,nhq->rhn", self.inverse.reshape(-1, h, q), targets)
        chosen = np.empty((len(self.axes), 0))
        for pool in (edits, per_step.reshape(len(self.axes), -1), np.eye(len(self.axes))):
            norms = np.linalg.norm(pool, axis=0)
            pool = pool[:, norms > 0] / norms[norms > 0]
            residual = pool - chosen @ (chosen.T @ pool)
            directions, singular, _ = np.linalg.svd(residual, full_matrices=False)
            count = min(rank - chosen.shape[1], int(np.sum(singular > 1e-8)))
            chosen = np.column_stack((chosen, directions[:, :count]))
            if chosen.shape[1] == rank:
                break
        chosen, _ = np.linalg.qr(chosen)
        return chosen.T @ self.axes, self.forward @ chosen
