# Decisions

- D01: A many-to-many concordance has structural edges separate from per-flow
  weights; all enabled edges determine emitted labels. A zero share remains an edge.
- D02: WP01 validates reconstruction only by original Eurostat parents and does not
  constrain the intentionally different final target classification.
- D03: Donor-block allocation does not execute RAS/GRAS, so its unused controls and
  fabricated zero-iteration balance result are removed.
