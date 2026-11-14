# Runpod setup and alternatives checked 2026-11-14

The user requested the official setup guide at https://agents.runpod.io/.
The guide was fetched directly after the web fetcher failed. Following its
Codex route, the official marketplace and `runpod@runpod` plugin were
installed. Verified version: **2.1.0**, installed and enabled; 17 skill
directories; MCP URL `https://mcp.runpod.io/`. No duplicate server was added.

An initial Git download failed because a pre-existing Git proxy pointed to
an inactive `127.0.0.1:7897`. A process-local `http.proxy=` override allowed
installation without changing the user's global Git settings.

OAuth login completed successfully with `codex mcp login runpod`; `codex mcp
list` reports the server enabled with OAuth. Live account inventory is not
yet verified because new Runpod tools are not exposed in the active session.
The guide says Codex
Desktop must be fully quit and reopened to expose newly installed tools.
No Pod has been created and no Runpod charge has been authorized or incurred.

## Public price options

The official pricing page, updated November 12, lists:

| GPU Pod | Community from/hour | Secure/hour | 2-hour GPU-only estimate |
|---|---:|---:|---:|
| A100 PCIe | $1.19 | $2.09 | $2.38–$4.18 |
| A100 SXM | $1.39 | $2.19 | $2.78–$4.38 |

Source: https://www.runpod.io/pricing (checked November 14).
These are public rates, **not confirmed account-specific inventory or a
reservation quote**. Region, GPU count and storage affect actual offers.
The advertised A100 PCIe listing generally refers to 80 GB; verify the exact
device before deployment. Lonestar6 uses A100 PCIe 40 GB, so do not pool
different GPU variants in a single matched comparison.

For this project use one dedicated on-demand GPU Pod, not a Serverless
inference worker or a MIG slice. A100 PCIe is the closest family match;
SXM is a separate hardware cohort. Secure Cloud is a reasonable default
when available. An isolated Pod still needs GPU/counter/software checks.

Storage is extra: the official page lists $0.10/GB/month for container disk
and running volume disk, $0.20/GB/month for stopped volume disk, and
$0.07/GB/month for standard network storage below 1 TB. Confirm the final
deployment quote and retrieve artifacts before deleting storage.

## Current decision

Lonestar6 job **3582290 actually started at 16:34:38 America/Chicago**,
earlier than its initial 18:27 estimate. Its A100 setup/pilot is underway.
Runpod is therefore an alternative, not needed just to eliminate the current
queue. Account-specific checking needs a restarted session with the new tools.

A read-only query to Runpod's public GraphQL catalog subsequently succeeded.
Its snapshot is in `reports/runpod_public_capacity.json`. It reported **Low**
Secure Cloud stock for A100 PCIe 80 GB ($2.09/hour) and A100 SXM 80 GB
($2.19/hour). Community stock fields were null, so they do not establish
deployable stock. A100 SXM 40 GB appears at $1.00/hour Community pricing,
with no confirmed stock and no Secure Cloud offering. These are temporary
catalog indicators, not successful reservations.

The initial model API blocker was `credit_balance_exhausted`. The user added
$20 credit, a diagnostic completion then succeeded, and the four-run local
real-model smoke comparison completed for approximately $0.0341.
