# Metz 170 Complete Filtered Package

This package contains the GPCNDTA/MEGDTA Metz 170-target dataset after removing exact pair overlaps with HMRLBA identity30 train+valid splits.

Counts:
- Original Metz pairs: 35,259
- Removed unique Metz pairs: 8
- Filtered pairs: 35251
- Drugs: 1423
- Targets: 170
- PDB structure files copied: 170

Important: `HMRLBA_raw_style/metz_170_clean` is a raw-style data layout. The current HMRLBA code still needs a Metz/multi-ligand dataset handler to consume these pairs directly.
