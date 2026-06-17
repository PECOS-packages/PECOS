# Inner-decoder study results

LER with Jeffreys 95% intervals (Beta(k+1/2, n-k+1/2)); seeds pooled into one
binomial per (family, d, p, inner). `k/n` = failures / shots.

## cx: distance suppression + ranking

### p = 0.002

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 1039/150000 6.93e-03 [6.5e-03,7.4e-03] | 294/150000 1.96e-03 [1.7e-03,2.2e-03] | 59/150000 3.93e-04 [3.0e-04,5.0e-04] |
| pecos_uf:bp | 1069/150000 7.13e-03 [6.7e-03,7.6e-03] | 556/150000 3.71e-03 [3.4e-03,4.0e-03] | 159/150000 1.06e-03 [9.0e-04,1.2e-03] |
| belief_matching | 1042/150000 6.95e-03 [6.5e-03,7.4e-03] | 294/150000 1.96e-03 [1.7e-03,2.2e-03] | 59/150000 3.93e-04 [3.0e-04,5.0e-04] |
| pymatching | -- | -- | -- |
| tesseract | -- | -- | -- |

- d=3: best **fusion_blossom_serial** 6.93e-03 vs pecos_uf:bp 7.13e-03 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 1.96e-03 vs pecos_uf:bp 3.71e-03 (1.9x) -- Jeffreys intervals DISJOINT
- d=7: best **fusion_blossom_serial** 3.93e-04 vs pecos_uf:bp 1.06e-03 (2.7x) -- Jeffreys intervals DISJOINT

### p = 0.003

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 2158/150000 1.44e-02 [1.4e-02,1.5e-02] | 1010/150000 6.73e-03 [6.3e-03,7.2e-03] | 333/150000 2.22e-03 [2.0e-03,2.5e-03] |
| pecos_uf:bp | 2217/150000 1.48e-02 [1.4e-02,1.5e-02] | 1641/150000 1.09e-02 [1.0e-02,1.1e-02] | 662/150000 4.41e-03 [4.1e-03,4.8e-03] |
| belief_matching | 2159/150000 1.44e-02 [1.4e-02,1.5e-02] | 1010/150000 6.73e-03 [6.3e-03,7.2e-03] | 333/150000 2.22e-03 [2.0e-03,2.5e-03] |
| pymatching | -- | -- | -- |
| tesseract | -- | -- | -- |

- d=3: best **fusion_blossom_serial** 1.44e-02 vs pecos_uf:bp 1.48e-02 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 6.73e-03 vs pecos_uf:bp 1.09e-02 (1.6x) -- Jeffreys intervals DISJOINT
- d=7: best **fusion_blossom_serial** 2.22e-03 vs pecos_uf:bp 4.41e-03 (2.0x) -- Jeffreys intervals DISJOINT

### p = 0.005

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 5711/150000 3.81e-02 [3.7e-02,3.9e-02] | 4316/150000 2.88e-02 [2.8e-02,3.0e-02] | 2636/150000 1.76e-02 [1.7e-02,1.8e-02] |
| pecos_uf:bp | 5803/150000 3.87e-02 [3.8e-02,4.0e-02] | 5967/150000 3.98e-02 [3.9e-02,4.1e-02] | 4051/150000 2.70e-02 [2.6e-02,2.8e-02] |
| belief_matching | 5730/150000 3.82e-02 [3.7e-02,3.9e-02] | 4316/150000 2.88e-02 [2.8e-02,3.0e-02] | 2636/150000 1.76e-02 [1.7e-02,1.8e-02] |
| pymatching | -- | -- | -- |
| tesseract | -- | -- | -- |

- d=3: best **fusion_blossom_serial** 3.81e-02 vs pecos_uf:bp 3.87e-02 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 2.88e-02 vs pecos_uf:bp 3.98e-02 (1.4x) -- Jeffreys intervals DISJOINT
- d=7: best **fusion_blossom_serial** 1.76e-02 vs pecos_uf:bp 2.70e-02 (1.5x) -- Jeffreys intervals DISJOINT

### cx: suppression exponent (LER ~ (p/p_th)^((d+1)/2))

- p=0.002 fusion_blossom_serial: suppresses; per-step LER ratio 3.5x, 5.0x
- p=0.002 pecos_uf:bp: suppresses; per-step LER ratio 1.9x, 3.5x
- p=0.002 belief_matching: suppresses; per-step LER ratio 3.5x, 5.0x
- p=0.003 fusion_blossom_serial: suppresses; per-step LER ratio 2.1x, 3.0x
- p=0.003 pecos_uf:bp: suppresses; per-step LER ratio 1.4x, 2.5x
- p=0.003 belief_matching: suppresses; per-step LER ratio 2.1x, 3.0x
- p=0.005 fusion_blossom_serial: suppresses; per-step LER ratio 1.3x, 1.6x
- p=0.005 pecos_uf:bp: NOT monotone; per-step LER ratio 1.0x, 1.5x
- p=0.005 belief_matching: suppresses; per-step LER ratio 1.3x, 1.6x

## memory: distance suppression + ranking

### p = 0.002

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 421/300000 1.40e-03 [1.3e-03,1.5e-03] | 111/300000 3.70e-04 [3.1e-04,4.4e-04] | 28/300000 9.33e-05 [6.3e-05,1.3e-04] |
| pecos_uf:bp | 425/300000 1.42e-03 [1.3e-03,1.6e-03] | 280/300000 9.33e-04 [8.3e-04,1.0e-03] | 62/300000 2.07e-04 [1.6e-04,2.6e-04] |
| belief_matching | 420/300000 1.40e-03 [1.3e-03,1.5e-03] | 111/300000 3.70e-04 [3.1e-04,4.4e-04] | 28/300000 9.33e-05 [6.3e-05,1.3e-04] |
| pymatching | 421/300000 1.40e-03 [1.3e-03,1.5e-03] | 111/300000 3.70e-04 [3.1e-04,4.4e-04] | 28/300000 9.33e-05 [6.3e-05,1.3e-04] |
| tesseract | 421/300000 1.40e-03 [1.3e-03,1.5e-03] | 111/300000 3.70e-04 [3.1e-04,4.4e-04] | 28/300000 9.33e-05 [6.3e-05,1.3e-04] |

- d=3: best **belief_matching** 1.40e-03 vs pecos_uf:bp 1.42e-03 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 3.70e-04 vs pecos_uf:bp 9.33e-04 (2.5x) -- Jeffreys intervals DISJOINT
- d=7: best **fusion_blossom_serial** 9.33e-05 vs pecos_uf:bp 2.07e-04 (2.2x) -- Jeffreys intervals DISJOINT

### p = 0.003

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 934/300000 3.11e-03 [2.9e-03,3.3e-03] | 375/300000 1.25e-03 [1.1e-03,1.4e-03] | 129/300000 4.30e-04 [3.6e-04,5.1e-04] |
| pecos_uf:bp | 944/300000 3.15e-03 [3.0e-03,3.4e-03] | 738/300000 2.46e-03 [2.3e-03,2.6e-03] | 233/300000 7.77e-04 [6.8e-04,8.8e-04] |
| belief_matching | 933/300000 3.11e-03 [2.9e-03,3.3e-03] | 375/300000 1.25e-03 [1.1e-03,1.4e-03] | 129/300000 4.30e-04 [3.6e-04,5.1e-04] |
| pymatching | 934/300000 3.11e-03 [2.9e-03,3.3e-03] | 376/300000 1.25e-03 [1.1e-03,1.4e-03] | 130/300000 4.33e-04 [3.6e-04,5.1e-04] |
| tesseract | 934/300000 3.11e-03 [2.9e-03,3.3e-03] | 376/300000 1.25e-03 [1.1e-03,1.4e-03] | 131/300000 4.37e-04 [3.7e-04,5.2e-04] |

- d=3: best **belief_matching** 3.11e-03 vs pecos_uf:bp 3.15e-03 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 1.25e-03 vs pecos_uf:bp 2.46e-03 (2.0x) -- Jeffreys intervals DISJOINT
- d=7: best **fusion_blossom_serial** 4.30e-04 vs pecos_uf:bp 7.77e-04 (1.8x) -- Jeffreys intervals DISJOINT

### p = 0.005

| inner | d=3 | d=5 | d=7 |
|---|---|---|---|
| fusion_blossom_serial | 2418/300000 8.06e-03 [7.7e-03,8.4e-03] | 1616/300000 5.39e-03 [5.1e-03,5.7e-03] | 830/300000 2.77e-03 [2.6e-03,3.0e-03] |
| pecos_uf:bp | 2481/300000 8.27e-03 [8.0e-03,8.6e-03] | 2625/300000 8.75e-03 [8.4e-03,9.1e-03] | 1519/300000 5.06e-03 [4.8e-03,5.3e-03] |
| belief_matching | 2429/300000 8.10e-03 [7.8e-03,8.4e-03] | 1616/300000 5.39e-03 [5.1e-03,5.7e-03] | 830/300000 2.77e-03 [2.6e-03,3.0e-03] |
| pymatching | 2415/300000 8.05e-03 [7.7e-03,8.4e-03] | 1618/300000 5.39e-03 [5.1e-03,5.7e-03] | 828/300000 2.76e-03 [2.6e-03,3.0e-03] |
| tesseract | 2416/300000 8.05e-03 [7.7e-03,8.4e-03] | 1619/300000 5.40e-03 [5.1e-03,5.7e-03] | 833/300000 2.78e-03 [2.6e-03,3.0e-03] |

- d=3: best **pymatching** 8.05e-03 vs pecos_uf:bp 8.27e-03 (1.0x) -- Jeffreys intervals overlap
- d=5: best **fusion_blossom_serial** 5.39e-03 vs pecos_uf:bp 8.75e-03 (1.6x) -- Jeffreys intervals DISJOINT
- d=7: best **pymatching** 2.76e-03 vs pecos_uf:bp 5.06e-03 (1.8x) -- Jeffreys intervals DISJOINT

### memory: suppression exponent (LER ~ (p/p_th)^((d+1)/2))

- p=0.002 fusion_blossom_serial: suppresses; per-step LER ratio 3.8x, 4.0x
- p=0.002 pecos_uf:bp: suppresses; per-step LER ratio 1.5x, 4.5x
- p=0.002 belief_matching: suppresses; per-step LER ratio 3.8x, 4.0x
- p=0.002 pymatching: suppresses; per-step LER ratio 3.8x, 4.0x
- p=0.002 tesseract: suppresses; per-step LER ratio 3.8x, 4.0x
- p=0.003 fusion_blossom_serial: suppresses; per-step LER ratio 2.5x, 2.9x
- p=0.003 pecos_uf:bp: suppresses; per-step LER ratio 1.3x, 3.2x
- p=0.003 belief_matching: suppresses; per-step LER ratio 2.5x, 2.9x
- p=0.003 pymatching: suppresses; per-step LER ratio 2.5x, 2.9x
- p=0.003 tesseract: suppresses; per-step LER ratio 2.5x, 2.9x
- p=0.005 fusion_blossom_serial: suppresses; per-step LER ratio 1.5x, 1.9x
- p=0.005 pecos_uf:bp: NOT monotone; per-step LER ratio 0.9x, 1.7x
- p=0.005 belief_matching: suppresses; per-step LER ratio 1.5x, 1.9x
- p=0.005 pymatching: suppresses; per-step LER ratio 1.5x, 2.0x
- p=0.005 tesseract: suppresses; per-step LER ratio 1.5x, 1.9x

## memory: threshold crossing

### fusion_blossom_serial

| p | d=3 | d=5 | d=7 |
|---|---|---|---|
| 0.004 | 4.82e-03 | 2.72e-03 | 1.44e-03 |
| 0.005 | 7.40e-03 | 5.16e-03 | 3.16e-03 |
| 0.006 | 1.13e-02 | 8.86e-03 | 5.90e-03 |
| 0.007 | 1.44e-02 | 1.36e-02 | 1.06e-02 |
| 0.008 | 1.87e-02 | 2.05e-02 | 1.68e-02 |
| 0.009 | 2.35e-02 | 2.77e-02 | 2.53e-02 |
| 0.010 | 2.85e-02 | 3.55e-02 | 3.69e-02 |
| 0.012 | 3.95e-02 | 5.50e-02 | 6.57e-02 |

- threshold estimate (d=7 stops beating d=3): ~0.009

### pecos_uf:bp

| p | d=3 | d=5 | d=7 |
|---|---|---|---|
| 0.004 | 4.88e-03 | 5.30e-03 | 2.52e-03 |
| 0.005 | 7.62e-03 | 9.02e-03 | 5.18e-03 |
| 0.006 | 1.14e-02 | 1.38e-02 | 9.64e-03 |
| 0.007 | 1.47e-02 | 1.96e-02 | 1.62e-02 |
| 0.008 | 1.94e-02 | 2.77e-02 | 2.50e-02 |
| 0.009 | 2.41e-02 | 3.69e-02 | 3.50e-02 |
| 0.010 | 2.92e-02 | 4.65e-02 | 5.00e-02 |
| 0.012 | 4.03e-02 | 6.90e-02 | 8.44e-02 |

- threshold estimate (d=7 stops beating d=3): ~0.007

### pymatching

| p | d=3 | d=5 | d=7 |
|---|---|---|---|
| 0.004 | 4.82e-03 | 2.74e-03 | 1.44e-03 |
| 0.005 | 7.40e-03 | 5.18e-03 | 3.16e-03 |
| 0.006 | 1.12e-02 | 8.86e-03 | 5.92e-03 |
| 0.007 | 1.44e-02 | 1.37e-02 | 1.06e-02 |
| 0.008 | 1.87e-02 | 2.04e-02 | 1.68e-02 |
| 0.009 | 2.36e-02 | 2.78e-02 | 2.53e-02 |
| 0.010 | 2.86e-02 | 3.55e-02 | 3.69e-02 |
| 0.012 | 3.94e-02 | 5.51e-02 | 6.58e-02 |

- threshold estimate (d=7 stops beating d=3): ~0.009

## speed (d=7, p=0.003, n per cell as sampled)

| family | inner | build ms | decode s | us/shot |
|---|---|---:|---:|---:|
| cx | fusion_blossom_serial | 55.1 | 61.22 | 1224.5 |
| cx | belief_matching | 234.4 | 140.59 | 2811.8 |
| cx | pecos_uf:bp | 228.1 | 356.84 | 7136.8 |
| memory | pymatching | 11.6 | 1.56 | 31.1 |
| memory | fusion_blossom_serial | 11.3 | 9.03 | 180.5 |
| memory | belief_matching | 27.7 | 24.13 | 482.5 |
| memory | tesseract | 16.9 | 44.44 | 888.8 |
| memory | pecos_uf:bp | 26.3 | 49.61 | 992.2 |
