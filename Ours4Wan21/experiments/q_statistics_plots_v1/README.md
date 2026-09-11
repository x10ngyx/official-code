# Q statistics plots

`plot.py` reads the frozen twelve-group post300 Q census and writes two 12-panel
PNG/SVG figures, a tidy CSV and source/validation metadata under the external
training readout's `q_statistics/` directory. No training or checkpoint selection changes.

Chart contract: trend / small-multiple line charts; one panel per group, 101 epochs
(300–400) per panel, common axes, no smoothing. Plot mean_Q and Q-IQR separately
from Q=min(Q1,Q2), pooled across both actions and all discretionary validation
states. Q-IQR uses the existing midpoint empirical quantile interpolation, not
an average of adjacent-epoch IQRs. The selected checkpoint is marked with an open
circle. Single blue root (#3569A8), neutral grid and labels, no color-only grouping.
Question: how do the Q location and spread evolve across the saved late-training
window? These are offline scale diagnostics, not a video-quality ranking.

Run using the Wan2.2 environment Python: `python plot.py`.
