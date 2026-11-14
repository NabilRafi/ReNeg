"""Hu et al., "Respecting Modality Gap in Post-hoc OOD Detection with Pre-trained VLMs" (ICML 2026), on cached
CLIP features: a reproduction ATTEMPT that does NOT reproduce the paper (docs/NEGATIVE_RESULTS.md, section 7).
No public code was found (Nov 2026); this follows the paper: Alg. 1 and Eqs. (3), (7)-(15).

  prototypes   w_y for the K ID classes and the L = 10,000 negative labels, initialised with their text embeddings
  gate         S_NegLabel(x) (Eq. 3, fixed text embeddings, tau): positive if >= beta, negative if <= 1 - beta
  update       one projected gradient step per sample on the pseudo-loss (Eq. 7, temperature kappa):
               positives move the ID prototypes, negatives the OOD prototypes; step rho / sqrt(count) (Eq. 11)
  score        ID mass under the updated prototypes (Eq. 14, tau)

Points the paper leaves open, fixed before any run (results/p5_other_bases/PREREGISTRATION_P5.md):
  - step size eta = rho / sqrt(c), the form of Eq. (8) (Eq. (12) prints 1 / eta);
  - soft pseudo-labels = CLIP's text softmax (tau) over the ID names for positives, over the negative labels for
    negatives; each branch's loss runs over its own label set.
One sample per step, as in the paper; ``step`` loops over the batch.

Result (Nov 14, 2026; 8-bit features, stream order 0; FPR95 OOD positive / ID positive): NINCO 73.79 / 64.33 with
the split loss and 76.43 / 67.29 with the union loss, against 50.60 in the paper; Places 55.17 / 50.18 (split)
against 28.55. Neither reading is the authors' method, so ReNeg was not put on top of it.
"""
from dataclasses import dataclass, replace
import torch
from oodlab.methods.base import StepOutput, StreamMethod


@dataclass
class HuConfig:
    tau: float = 0.01
    kappa: float = 0.05
    rho: float = 0.1
    beta: float = 0.95
    label_space: str = "split"     # "split": each branch's loss over its own labels (first reading, pre-registered)
                                   # "union": one loss over ID + OOD pseudo-classes (the literal Eq. 7), each branch
                                   #          updating only its own prototypes (Eqs. 12-13)

    def with_(self, **kw):
        return replace(self, **kw)


class HuFeat(StreamMethod):
    name = "hu"

    def __init__(self, id_text, neg_text, cfg=None, device="cpu"):
        self.cfg = cfg or HuConfig()
        self.R_id = torch.as_tensor(id_text).float().to(device)
        self.R_neg = torch.as_tensor(neg_text).float().to(device)
        self.C = self.R_id.shape[0]
        self.reset()

    def reset(self):
        self.W_id = self.R_id.clone()
        self.W_neg = self.R_neg.clone()
        self.c_pos = 0
        self.c_neg = 0

    def _move(self, W, u, a, z):
        """W <- l2(W - a z^T) for unit rows and unit z; returns the new W and the new W @ z in closed form."""
        n = torch.sqrt((1 - 2 * a * u + a * a).clamp_min(1e-12))
        return (W - a[:, None] * z[None, :]) / n[:, None], (u - a) / n

    @torch.no_grad()
    def step(self, feats):
        c, C = self.cfg, self.C
        Z = feats.float()
        Z = Z / Z.norm(dim=1, keepdim=True)
        lt_id = (Z @ self.R_id.T) / c.tau                      # fixed text side, whole batch
        lt_neg = (Z @ self.R_neg.T) / c.tau
        s_nl = torch.sigmoid(torch.logsumexp(lt_id, 1) - torch.logsumexp(lt_neg, 1))   # Eq. (3)
        union = c.label_space == "union"
        if union:
            q_all = torch.softmax(torch.cat([lt_id, lt_neg], 1), 1)
            q_id, q_neg = q_all[:, :C], q_all[:, C:]
        else:
            q_id, q_neg = torch.softmax(lt_id, 1), torch.softmax(lt_neg, 1)
        pos = (s_nl >= c.beta).tolist()
        neg = (s_nl <= 1 - c.beta).tolist()
        out = torch.empty(len(Z))
        for i in range(len(Z)):
            z = Z[i]
            u_id, u_neg = self.W_id @ z, self.W_neg @ z
            if pos[i] or neg[i]:
                if union:
                    pi = torch.softmax(torch.cat([u_id, u_neg]) / c.kappa, 0)
                    pi_id, pi_neg = pi[:C], pi[C:]
                elif pos[i]:
                    pi_id = torch.softmax(u_id / c.kappa, 0)
                else:
                    pi_neg = torch.softmax(u_neg / c.kappa, 0)
                if pos[i]:
                    self.c_pos += 1
                    a = (c.rho / self.c_pos ** 0.5) * (pi_id - q_id[i]) / c.kappa
                    self.W_id, u_id = self._move(self.W_id, u_id, a, z)
                else:
                    self.c_neg += 1
                    a = (c.rho / self.c_neg ** 0.5) * (pi_neg - q_neg[i]) / c.kappa
                    self.W_neg, u_neg = self._move(self.W_neg, u_neg, a, z)
            out[i] = torch.sigmoid(torch.logsumexp(u_id / c.tau, 0) - torch.logsumexp(u_neg / c.tau, 0))   # Eq. (14)
        pred = (Z @ self.W_id.T).argmax(1)
        return StepOutput(pred=pred, score=out, info={"c_pos": self.c_pos, "c_neg": self.c_neg, "s_nl": s_nl})
