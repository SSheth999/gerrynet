# GerryNet

Graph Neural Network (GNN) experiments for detecting gerrymandering in U.S.
congressional district plans.

## Repo layout

```
.
├── data/                 # raw + processed data (gitignored)
├── graph/                # graph construction (adjacency, node features)
├── scripts/              # data download / preprocessing entry points
└── gerrynet-frontend/    # Next.js UI for browsing maps and predictions
```

## Data

`scripts/download_shapefiles.py` pulls Congressional District TIGER/Line
shapefiles from the U.S. Census Bureau into `data/raw/shapefiles/`:

| Plan  | Vintage   | Layout     |
| ----- | --------- | ---------- |
| cd113 | TIGER2013 | national   |
| cd115 | TIGER2016 | national   |
| cd116 | TIGER2020 | national   |
| cd118 | TIGER2023 | per-state  |
| cd119 | TIGER2025 | per-state  |

Run:

```bash
python3 scripts/download_shapefiles.py            # all plans
python3 scripts/download_shapefiles.py cd118 cd119  # subset
```

The script is idempotent (skips already-extracted files) and retries on
transient Cloudflare errors.

## Frontend

```bash
cd gerrynet-frontend
npm install
npm run dev
```
