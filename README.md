# UO Course Enrollment Dashboard

An interactive dashboard for exploring University of Oregon course enrollment
from Fall 1990 through the present academic year. Enrollment data is scraped
from the DuckWeb class schedule; course descriptions come from the 2025-26
course catalog.

## What the dashboard does

- Pick any (Subject, Course Number) pair and add it to a plot of enrollment
  vs. time. Add multiple courses to compare.
- Choose a year range, whether to include summer, how to handle multiple
  sections, and how to filter by section format.
- Optionally normalize each series to its first non-zero term so you can
  compare relative changes regardless of absolute scale.
- Optionally sum each course's enrollment by academic year and plot it as
  a horizontal step that spans the AY (Fall through Spring, or through
  Summer if summer is included).
- Each course is drawn with a distinct color *and* marker symbol so series
  remain distinguishable in print and for some kinds of color-vision
  differences. Assignments are deterministic, so a course gets the same
  color and marker every session.
- When 5 or fewer courses are plotted, the 2025-26 catalog descriptions
  appear below the plot.
- Export the underlying data as CSV; export the plot as PNG via the camera
  icon in the plot toolbar.

## Decisions and limitations

The data has known sharp edges. The dashboard makes them visible rather than
trying to fix them:

- **Course-number changes are not reconciled.** If a course was renumbered
  partway through its history (e.g. PHYS 101 → PHYS 121), each number is
  treated as a separate course. The resulting time series will be jagged.
  Plot both numbers if you suspect a renumbering. *Exception:* the
  "Combine '…Z' transferable variants" option (on by default) merges
  courses that differ only by a trailing "Z" — UO has been appending the
  Z suffix to many introductory courses (CH 221 → CH 221Z, BI 221 → BI
  221Z, etc.) to mark statewide-transferable equivalents, and these are
  effectively the same course. Other letter suffixes (H = honors, L =
  lab, M, N) are always kept separate.
- **Subject-code changes** are reconciled via `modifications.csv` at the
  project root. Each `merge, NEW, OLD` line tells the dashboard that the
  two codes are the same department — for example `merge, ERTH, GEOL`
  collapses old GEOL data into ERTH. Both codes still appear in the
  subject dropdown, each annotated with the other (e.g. `ERTH (+GEOL)`
  and `GEOL (+ERTH)`), so you can find courses under either name; either
  picks the same underlying canonical subject. Edit `modifications.csv`
  and restart (or clear the Streamlit cache) to add or remove merges.
- **Cross-listed courses are not merged.** A course offered under two
  subject codes appears as two independent series. The dashboard does not
  attempt to detect or combine cross-listings.
- **Format filtering.** The DuckWeb data labels each section with a Format
  (`Lecture`, `Lab`, `Tutorial`, `Discussion`, `Seminar`, ...). Some
  sections — typically the primary listing — have no explicit format
  label; the dashboard treats those as lectures. The "Lectures only" filter
  includes all rows with normalized format `Lecture`. "First section only"
  keeps only the first row listed for each (Term, Subject, Course Number)
  group, which is usually the lecture but is robust to format-labeling
  inconsistencies across years. "All sections" includes labs, discussions,
  etc. and will overcount students if a course has multiple non-lecture
  components.
- **Enrollment values are snapshots.** Past terms reflect final enrollment
  (Max − Avail at scrape time); current/future terms reflect the moment of
  the scrape and may change.

## How time is plotted

The x-axis is a continuous "calendar year + offset" value:

| Quarter | Offset | Example (academic year 2025) |
|---------|--------|------------------------------|
| Fall    | 0.87   | Fall 2025 → 2025.87 |
| Winter  | 0.125  | Winter 2026 → 2026.125 |
| Spring  | 0.375  | Spring 2026 → 2026.375 |
| Summer  | 0.625  | Summer 2026 → 2026.625 |

Dashed vertical lines mark the boundary of each new academic year, at
calendar year + 0.75 (≈ October 1).

## Project layout

```
uo-enrollment-dashboard/
├── app.py                 Streamlit app
├── prepare_data.py        Builds data/enrollment.parquet from combined.csv
├── modifications.csv      Subject-code merge rules (editable)
├── requirements.txt
├── README.md              (this file — also displayed in the app)
├── .gitignore
└── data/
    ├── enrollment.parquet   Pre-processed enrollment, committed
    └── catalog.csv          2025-26 catalog, committed
```

## Building the data

You need two source files from the scraper projects:

- `combined.csv` produced by `parse_schedules.py` (one row per course
  section per term, ~1990 onward).
- `catalog.csv` produced by `scrape_catalog.py` (2025-26 catalog
  descriptions).

Run the prep step once:

```bash
python prepare_data.py \
    --combined /path/to/uo_schedules/csv/combined.csv \
    --catalog  /path/to/catalog.csv
```

This writes `data/enrollment.parquet` and copies `data/catalog.csv`. The
parquet keeps only the columns the dashboard needs and is much smaller
than the raw CSV. Commit the contents of `data/`.

When new terms become available, re-scrape, re-run `prepare_data.py`, and
push.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private — Streamlit Cloud supports
   both for free accounts).
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click
   **Create app**.
3. Select the repo, branch (usually `main`), and `app.py` as the main file.
   Pick a Python version (3.11 or 3.12 both work).
4. Click **Deploy**. The first build takes a few minutes; subsequent pushes
   redeploy automatically.

No secrets are required.

### A note on data size

`enrollment.parquet` is typically 10–30 MB for the full 1990-present range,
well under GitHub's 50 MB-per-file warning threshold and Streamlit Cloud's
memory limits. If it ever exceeds 50 MB, look at dropping the `Instructors`
column (already not included) or switching to `compression="zstd"` in
`prepare_data.py`.

## Source code for the scrapers

The scrapers that produce `combined.csv` and `catalog.csv` live in separate
repos:

- DuckWeb schedule scraper → `download_schedules.py` + `parse_schedules.py`
- Catalog scraper → `scrape_catalog.py`

This dashboard repo only consumes their output.
