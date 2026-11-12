"""Off-by-default flags for the ANVIL cascade (optim/anvil.py), and the alternative map schedule. No torch / triton
import here, so the CPU tools (182/optim-cost/) can read it.

  OPTIM_MOEFUSED=1  the expert bank (moe_bank) runs the fused tail of the other banks (anvil_opt.pre_fused + the fused
                    cascade) at its own margin 1.1, instead of the inductor-compiled anvil_cascade. Same arithmetic
                    (intended bitwise equal; the GPU test 182/optim-cost/test_optim_cost.py checks it on the node).
                    Time only.
  OPTIM_MAPS=3      three quintic maps (ANVIL_MAPS3) instead of the four ANVIL_MAPS: one Gram / polynomial / update round
                    less per bank per step. Changes the update: judge at equal loss (step sweep).
  OPTIM_NORM=s4     normalize the cascade input by its Schatten-4 norm ||X^T X||_F^(1/2) instead of ||X||_F (both from
                    the first Gram, which the cascade computes anyway). Changes the update: judge at equal loss.
  OPTIM_PROFILE=1   CUDA-event timing of each label's wait and update inside AnvilAndAdam.step (optim/step_profile.py);
                    prints an `[optim-profile]` line every 100 steps. Diagnostic: changes no number.

OPTIM_MAPS=3 and OPTIM_NORM=s4 run through the fused cascade for every bank, so they imply OPTIM_MOEFUSED's expert-bank
path; compare them against the `optmoefused` arm for the time attribution.
"""
import os

# Designed like ANVIL_MAPS (182/optim-cost/README.md): minimax max|p(s) - 1| over normalized singular values
# s in [L, 1/1.05], composite of three odd quintics p(s) = a s + b s^3 + c s^5, bounded (< 5) up to s = 1.02. The shipped
# four maps reach max|p - 1| = 0.2388 on [3e-3, 1/1.05]; three maps reach the same 0.2387 only from L = 1.134e-2.
ANVIL_MAPS3 = [
    (7.980480337776, -22.11853355926, 15.541513802055),
    (3.471776759428, -2.517242228745, 0.483333382866),
    (2.629053639001, -2.009063427137, 0.482288148922),
]


def _env(name: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, "").strip().lower()
    assert value in allowed, f"{name}={value!r}: use one of {allowed}"
    return value


OPTIM_MOEFUSED = _env("OPTIM_MOEFUSED", ("", "0", "1")) == "1"
OPTIM_MAPS = int(_env("OPTIM_MAPS", ("", "4", "3")) or 4)
OPTIM_NORM_S4 = _env("OPTIM_NORM", ("", "fro", "s4")) == "s4"
OPTIM_PROFILE = _env("OPTIM_PROFILE", ("", "0", "1")) == "1"
# Any flag that changes how a bank update runs (OPTIM_PROFILE does not).
OPTIM_FLAGGED_UPDATE = OPTIM_MOEFUSED or OPTIM_MAPS != 4 or OPTIM_NORM_S4


def describe() -> str:
    """One line for the log."""
    return (f"[optim] moefused={int(OPTIM_MOEFUSED)} maps={OPTIM_MAPS} norm={'s4' if OPTIM_NORM_S4 else 'fro'} "
            f"profile={int(OPTIM_PROFILE)}")
