# Mixed3500 cached Q statistics

plot.py reads conservative/aggressive post300_validation_values.npz, verifies same validation row indices and epoch300–400, calculates mean of Q=min(Q1,Q2) over all states and both actions, midpoint empirical P25/P75 and IQR. Exports two comparison PNG/SVG files, CSV and source hash validation to suite/analysis/q_statistics/. Selected epochs374/358 marked. No inference, GPU work or extrapolation to epochs1–299.
