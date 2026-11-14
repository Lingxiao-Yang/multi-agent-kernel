# A100 Compute Options (checked 2026-11-14)

The primary A100 platform is TACC Lonestar6. We evaluated Runpod as a backup
in case the Lonestar6 queue delayed the study. No Runpod resources were
created and nothing was charged.

## Public Runpod pricing

From the official pricing page (updated 2026-11-12, checked 2026-11-14):

| GPU Pod | Community ($/h, from) | Secure ($/h) | 2-hour GPU-only estimate |
| --- | ---: | ---: | ---: |
| A100 PCIe | $1.19 | $2.09 | $2.38–$4.18 |
| A100 SXM | $1.39 | $2.19 | $2.78–$4.38 |

These are public list prices. They are not quotes for a specific account or
confirmations of availability; region, GPU count and storage change the
actual offer.

**Storage costs (per GB per month).**

| Storage type | Price |
| --- | ---: |
| Container disk | $0.10 |
| Running volume disk | $0.10 |
| Stopped volume disk | $0.20 |
| Network storage under 1 TB | $0.07 |

Source: <https://www.runpod.io/pricing>.

## Catalog snapshot

A read-only query of Runpod's public GraphQL catalog is saved in
[`runpod_public_capacity.json`](runpod_public_capacity.json).

| GPU | Secure Cloud stock | Price |
| --- | --- | ---: |
| A100 PCIe 80 GB | Low | $2.09/h |
| A100 SXM 80 GB | Low | $2.19/h |
| A100 SXM 40 GB | Not offered | $1.00/h (Community) |

- Community stock fields were empty for all of these GPUs, so they do not
  show any capacity that could be deployed.
- No Community stock was confirmed for the A100 SXM 40 GB.
- The catalog changes over time; a listing is not a reservation.

## Hardware-matching guidance

If Runpod is used, it should be:

- one dedicated on-demand GPU Pod;
- not a Serverless worker or a MIG slice.

A100 PCIe is the closest match to Lonestar6. Lonestar6 uses the A100 PCIe
40 GB, while Runpod's A100 PCIe listing is usually the 80 GB part, so check
the exact device first. Treat A100 SXM as a separate hardware cohort. Never
pool different GPU variants in one matched comparison.

Any new machine needs the same GPU, hardware-counter and software checks as
Lonestar6.

## Decision

Lonestar6 job 3582290 started at 16:34:38 America/Chicago, ahead of the
scheduler's 18:27 estimate. Runpod therefore stays a fallback and is not
needed to get past the queue.
