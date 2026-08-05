# LOGOS run-manifest summary

Cost model: FLOPs = 6*N_nonemb*D; H100 bf16 peak 989 TFLOP/s; MFU 0.38 bf16 / 0.30 quantized (~20% QAT penalty); $2.80/GPU-h. No overhead multiplier, so estimates sit at/below the plan's ranges (plan s12 carries ~+/-30% bars and +20% overhead on P2).

| Phase | Runs | Est GPU-h | Est cost | Plan s12 GPU-h | Plan s12 cost |
|-------|------|-----------|----------|----------------|---------------|
| p0 | 16 | 41 | $115 | 60-80 | $200-$300 |
| p1 | 44 | 210 | $587 | 220-260 | $700-$1,000 |
| p2 | 130 | 1,509 | $4,224 | 1800-1800 | $5,000-$6,000 |
| p2ext | 3 | 1,201 | $3,364 | 1150-1150 | $3,200-$3,200 |
| p3 | 6 | 1,547 | $4,332 | 1050-1450 | $3,000-$4,500 |
| p4 | 9 | 245 | $685 | 350-500 | $1,000-$1,500 |
| p5 | 1 | 2,538 | $7,106 | 2900-2900 | $8,000-$9,500 |
| **total** | **209** | **7,290** | **$20,412** | ~6,500-7,000 core | ~$21,000-26,000 core |

Reused runs (p2 125m rows, tag `reused_from_p1`) are emitted at zero cost.
