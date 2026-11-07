"""FP8_OFF=lm: the training loss with a bf16 lm_head GEMM, in plain torch (182/hadamard-fp8).

The upper-bound arm of the bf16-vs-fp8 test. It computes what FusedSoftcappedCrossEntropy /
SampledSoftcappedCrossEntropy compute -- softcap z = A * sigmoid((logit + B) / C), multi-token CE with
weights mtp_weights over targets t, t+1, ..., plus the prefix-token CE at prefix_weight -- with:
  - bf16 logits (bf16 GEMM, fp32 accumulate) instead of e4m3 logit bytes, and fp32 sigmoid / exp
    instead of the kernel's fp16 sigmoid cache;
  - an fp32 logit gradient and bf16 gradient GEMMs instead of e5m2 / fp8.
The kernel's gradient convention is kept exactly: the softmax term is scaled by
S_w = sum(mtp_weights) + prefix_weight * [prefix target valid] on every row (the kernel does not drop
the MTP weights whose target falls past the end of the batch), and the one-hot terms only count valid
targets. Rows are processed in slabs, recomputing the logits in the backward, so no [T, V] tensor lives
across the forward. Speed is not a goal.
"""
import torch

# Must equal perf/kernels/cross_entropy.py's SOFTCAP_A/B/C (model/gpt.py asserts it).
SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5
SLAB_ROWS = 4096


def _shifted_targets(targets: torch.Tensor, n_predict: int) -> torch.Tensor:
    """[n, n_predict]: column k is targets[i + k], -1 past the end."""
    n = targets.shape[0]
    cols = [targets] + [torch.cat((targets[k:], targets.new_full((k,), -1))) for k in range(1, n_predict)]
    return torch.stack(cols[:n_predict], dim=1).view(n, n_predict)


def _softcap(logits_bf16: torch.Tensor):
    s = torch.sigmoid((logits_bf16.float() + SOFTCAP_B) / SOFTCAP_C)
    return SOFTCAP_A * s, s


def _slab_loss_terms(z, tgt, valid, ptgt, pvalid, mtp_weights, prefix_weight):
    lse = torch.logsumexp(z, dim=1)
    z_t = z.gather(1, tgt.clamp(min=0))
    loss = ((lse[:, None] - z_t) * (mtp_weights[None, :] * valid)).sum(dim=1)
    z_p = z.gather(1, ptgt.clamp(min=0)[:, None])[:, 0]
    loss = loss + prefix_weight * pvalid * (lse - z_p)
    return loss, lse


class BF16SoftcappedCrossEntropy(torch.autograd.Function):
    """Per-token training loss from x [n, D] and a [D, V'] bf16 lm_head (V' = the vocab or the sampled
    candidate set). targets / prefix_targets index columns of W (class ids, or positions in C)."""
    @staticmethod
    def forward(ctx, x, W, targets, mtp_weights, prefix_targets, prefix_weight):
        n, V = x.shape[0], W.shape[1]
        mtp_weights = mtp_weights.float()
        pw = prefix_weight.reshape(()).float()
        tgt = _shifted_targets(targets, mtp_weights.shape[0])
        valid = ((tgt >= 0) & (tgt < V)).float()
        pvalid = ((prefix_targets >= 0) & (prefix_targets < V)).float()
        losses = []
        for lo in range(0, n, SLAB_ROWS):
            hi = min(lo + SLAB_ROWS, n)
            z, _ = _softcap(x[lo:hi] @ W)
            loss, _ = _slab_loss_terms(z, tgt[lo:hi], valid[lo:hi], prefix_targets[lo:hi], pvalid[lo:hi], mtp_weights, pw)
            losses.append(loss)
        ctx.save_for_backward(x, W, tgt, valid, prefix_targets, pvalid, mtp_weights, pw)
        return torch.cat(losses)

    @staticmethod
    def backward(ctx, grad_output):
        x, W, tgt, valid, ptgt, pvalid, mtp_weights, pw = ctx.saved_tensors
        n = x.shape[0]
        # The kernel's normalizer weight: every MTP weight, plus the prefix weight where a prefix exists.
        s_w_rows = mtp_weights.sum() + pw * (ptgt >= 0).float()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros(W.shape, dtype=torch.float32, device=W.device)
        for lo in range(0, n, SLAB_ROWS):
            hi = min(lo + SLAB_ROWS, n)
            xs = x[lo:hi]
            z, s = _softcap(xs @ W)
            lse = torch.logsumexp(z, dim=1, keepdim=True)
            grad_z = s_w_rows[lo:hi, None] * torch.exp(z - lse)
            grad_z = grad_z.scatter_add(1, tgt[lo:hi].clamp(min=0), -(mtp_weights[None, :] * valid[lo:hi]))
            grad_z = grad_z.scatter_add(1, ptgt[lo:hi].clamp(min=0)[:, None], -(pw * pvalid[lo:hi])[:, None])
            dlogits = (grad_z * ((SOFTCAP_A / SOFTCAP_C) * s * (1.0 - s)) * grad_output[lo:hi, None].float()).to(x.dtype)
            grad_x[lo:hi] = dlogits @ W.T
            grad_w += (xs.T @ dlogits).float()
        return grad_x, grad_w.to(W.dtype), None, None, None, None


def bf16_training_loss(x, lm_head_weight, targets, mtp_weights, prefix_targets, prefix_weight, candidates=None):
    """Full softmax (candidates None) or the sampled softmax over `candidates` (ascending class ids, with
    targets / prefix_targets already positions in it). The candidate columns are an index_select of the
    live bf16 lm_head, so its gradient comes back dense with zero non-candidate columns, as the fp8
    path's densify writes it."""
    W = lm_head_weight if candidates is None else lm_head_weight.index_select(1, candidates)
    return BF16SoftcappedCrossEntropy.apply(x, W.type_as(x), targets, mtp_weights, prefix_targets, prefix_weight)
