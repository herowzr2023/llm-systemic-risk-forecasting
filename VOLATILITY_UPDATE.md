# Volatility replacement — 2026-09-29

Volatility in the raw input and VIX in derived datasets now use the supplied RV_20D_Annualized_ddof1 series, matched by date. VIX is retained only as a legacy column name: the supplied series is labelled 20-day annualized realized volatility, not an implied-volatility index. The source calculation itself was not independently reconstructed.

Only volatility values changed; dates, row counts, Close, Log_Return, Market_Return and all other fields were preserved. Existing CS corrections were preserved. Historical completed training snapshots and prediction/DM results were not modified or rerun. New corrected h1/h5 inputs remain untrained.

Source SHA256: 521627d0ffa099b3d4c5b3e5fd38ea1a5c7107712410dba45eb511f3789cae8c
