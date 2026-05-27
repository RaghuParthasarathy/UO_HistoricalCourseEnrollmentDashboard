"""
UO Course Enrollment Dashboard

Streamlit app for visualizing enrollment over time for selected UO courses,
backed by data scraped from DuckWeb (1990–2025 academic years) and the
2025-26 course catalog.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, then point Streamlit Community Cloud at this
                repo and `app.py`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
ENROLLMENT_PARQUET = DATA_DIR / "enrollment.parquet"
CATALOG_CSV = DATA_DIR / "catalog.csv"
README_PATH = Path(__file__).parent / "README.md"
MODIFICATIONS_CSV = Path(__file__).parent / "modifications.csv"

# Quarter codes in 6-digit term codes:
#   01 = Fall, 02 = Winter, 03 = Spring, 04 = Summer
# The YYYY portion of the term code is the *academic year* — so
# term 202502 means Winter 2026 (academic year 2025-26).
QUARTER_INFO = {
    "01": {"name": "Fall",   "calendar_offset": 0, "axis_frac": 0.87},
    "02": {"name": "Winter", "calendar_offset": 1, "axis_frac": 0.125},
    "03": {"name": "Spring", "calendar_offset": 1, "axis_frac": 0.375},
    "04": {"name": "Summer", "calendar_offset": 1, "axis_frac": 0.625},
}

# Color palette used to assign one color per (Subject, Course_Number) line.
# Plotly's "Dark24" is a 24-color set with reasonable separation.
import plotly.express as px
COLOR_PALETTE = px.colors.qualitative.Dark24

# Cycled per-series so courses are distinguishable beyond color alone (useful
# for B&W printing and for some kinds of color-vision differences). The list
# alternates filled and outlined shapes, ordered to feel reasonably distinct
# at small sizes.
MARKER_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "pentagon", "hexagon",
    "star", "triangle-down",
    "circle-open", "square-open", "diamond-open", "triangle-up-open",
    "pentagon-open", "hexagon-open", "star-open", "triangle-down-open",
    "cross", "x",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_enrollment() -> pd.DataFrame:
    df = pd.read_parquet(ENROLLMENT_PARQUET)

    # Course_Number_Norm strips a single trailing "Z" so that courses like
    # "CH 221" and "CH 221Z" (the new state-transferable variant) can be
    # combined when the user enables the merge option. Other suffixes
    # (H = honors, L = lab, M, N) are preserved because they mark distinct
    # offerings.
    df["Course_Number_Norm"] = df["Course_Number"].str.replace(
        r"Z$", "", regex=True
    )

    # Derive year/quarter/x-axis columns.
    df["academic_year"] = df["Term_Code"].str[:4].astype(int)
    df["quarter_code"] = df["Term_Code"].str[4:6]
    df["quarter"] = df["quarter_code"].map(
        {k: v["name"] for k, v in QUARTER_INFO.items()}
    )
    df["calendar_year"] = df["academic_year"] + df["quarter_code"].map(
        {k: v["calendar_offset"] for k, v in QUARTER_INFO.items()}
    )
    df["axis_x"] = df["calendar_year"] + df["quarter_code"].map(
        {k: v["axis_frac"] for k, v in QUARTER_INFO.items()}
    )
    return df


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_CSV, dtype=str)
    df["Course_Number_Norm"] = df["Course_Number"].str.replace(
        r"Z$", "", regex=True
    )
    return df


def load_merge_rules(path: Path = MODIFICATIONS_CSV) -> dict[str, str]:
    """Read subject-merge rules from a small CSV.

    File format (no header; whitespace is stripped):
        merge, NEW_CODE, OLD_CODE

    For example:
        merge, ERTH, GEOL     # ERTH was previously GEOL
        merge, J, JCOM

    Returns {raw_subject: canonical_subject}. The canonical for a row is
    the *new* code; both the new and old map to it, so canonical lookup
    is idempotent. Unknown subjects map to themselves at query time.

    Not cached — the file is tiny and the user may edit it between runs.
    """
    if not path.exists():
        return {}
    rules: dict[str, str] = {}
    import csv
    with open(path, newline="") as f:
        for row in csv.reader(f):
            cells = [c.strip() for c in row]
            cells = [c for c in cells if c]  # drop empties
            if len(cells) < 3:
                continue
            if cells[0].lower() != "merge":
                continue
            new, old = cells[1], cells[2]
            rules[old] = new
            rules[new] = new
    return rules


def canonical_subject(subj: str, rules: dict[str, str]) -> str:
    return rules.get(subj, subj)


def build_subject_options(enrollment: pd.DataFrame,
                          rules: dict[str, str]) -> list[tuple[str, str]]:
    """Build the (display_label, canonical) pairs that populate the
    subject dropdown.

    Every alias declared in `modifications.csv` shows up as its own
    dropdown entry — annotated with the other aliases — *as long as the
    canonical has any data*. So when GEOL was renamed to ERTH and the
    current scrape only contains GEOL rows, we still show both
    `GEOL (+ERTH)` and `ERTH (+GEOL)` so users can find the course
    under either name.

    Returns the list sorted alphabetically by the raw subject code so
    the dropdown order is predictable.
    """
    raw_subjects_in_data = set(enrollment["Subject"].unique())
    # Set of canonicals that have any data at all.
    canonical_with_data = {canonical_subject(s, rules) for s in raw_subjects_in_data}

    # Aliases per canonical, taken from the rules table (not from data),
    # so a freshly renamed subject still shows both forms in the dropdown.
    canonical_to_aliases: dict[str, set[str]] = {}
    for raw, canon in rules.items():
        canonical_to_aliases.setdefault(canon, set()).add(raw)

    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for canon in canonical_with_data:
        aliases = canonical_to_aliases.get(canon)
        if aliases:
            for s in sorted(aliases):
                if s in seen:
                    continue
                seen.add(s)
                others = sorted(a for a in aliases if a != s)
                label = f"{s} (+{', '.join(others)})"
                options.append((label, canon))
        else:
            # No merge rule applies: a plain subject from the data.
            if canon not in seen:
                seen.add(canon)
                options.append((canon, canon))
    options.sort(key=lambda lc: lc[0].split(" ")[0])
    return options


@st.cache_data(show_spinner=False)
def subject_course_index(enrollment: pd.DataFrame,
                         num_col: str = "Course_Number",
                         rules: dict[str, str] | None = None
                         ) -> dict[str, list[str]]:
    """{ canonical_subject: [course_number, ...] } with courses naturally sorted.

    Pass num_col='Course_Number_Norm' to collapse Z-suffix variants.
    Pass `rules` to merge old/new subject codes (e.g. GEOL → ERTH); the
    returned keys are the *canonical* subjects, with course numbers
    drawn from the union of all merged raw subjects.
    """
    import re
    rules = rules or {}
    def natural_key(s: str):
        # Sort "101", "101H", "199", "199A", "201" sensibly.
        return [int(t) if t.isdigit() else t
                for t in re.split(r"(\d+)", s)]
    canon = enrollment["Subject"].map(lambda s: canonical_subject(s, rules))
    out: dict[str, list[str]] = {}
    for c, group in enrollment.assign(_canon=canon).groupby("_canon", observed=True):
        out[str(c)] = sorted(group[num_col].unique(), key=natural_key)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def course_label(subject: str, number: str) -> str:
    return f"{subject} {number}"


def stable_color(label: str) -> str:
    """Deterministic color assignment so a course is the same color every
    run, including across Streamlit reloads."""
    import zlib
    return COLOR_PALETTE[zlib.crc32(label.encode("utf-8")) % len(COLOR_PALETTE)]


def stable_marker(label: str) -> str:
    """Deterministic marker shape. Uses a different salt from stable_color
    so that color and marker indices don't move in lockstep."""
    import zlib
    return MARKER_SYMBOLS[zlib.crc32(b"marker:" + label.encode("utf-8"))
                          % len(MARKER_SYMBOLS)]


def filter_and_aggregate(
    enrollment: pd.DataFrame,
    selected: list[tuple[str, str]],
    format_mode: str,
    aggregate: str,
    year_start: int,
    year_end: int,
    exclude_summer: bool,
    merge_z: bool = False,
    rules: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply all the user's filters and produce plotting-ready rows.

    Returned columns:
      Subject, Course_Number, Course_Name, Term_Code, axis_x, calendar_year,
      academic_year, quarter, Format_Norm, Enrollment, series_key, label
    """
    if not selected:
        return enrollment.iloc[0:0].assign(
            series_key=pd.Series(dtype=str), label=pd.Series(dtype=str)
        )

    rules = rules or {}

    # --- Subject canonicalization (e.g. GEOL → ERTH per modifications.csv) ---
    # We add a Subject_Canonical column on the data and convert the selected
    # pairs to canonical form for the join. After the subset, we replace
    # Subject with Subject_Canonical so all downstream groupings collapse
    # the merged subjects into a single series identity.
    enrollment_local = enrollment
    if rules:
        canon_map = lambda s: rules.get(s, s)
        enrollment_local = enrollment.assign(
            Subject_Canonical=enrollment["Subject"].map(canon_map)
        )
        subject_col = "Subject_Canonical"
        selected_canon = [(canon_map(s), n) for s, n in selected]
    else:
        subject_col = "Subject"
        selected_canon = list(selected)

    # When merging Z-variants we match on the normalized column and
    # normalize the user's selected pairs so toggling the option doesn't
    # invalidate the selection list.
    import re
    num_col = "Course_Number_Norm" if merge_z else "Course_Number"
    if merge_z:
        selected_lookup = [(s, re.sub(r"Z$", "", n)) for s, n in selected_canon]
    else:
        selected_lookup = selected_canon

    # Subset to selected (canonical_subject, course_number) pairs.
    keys = pd.MultiIndex.from_tuples(selected_lookup,
                                     names=[subject_col, num_col])
    df = enrollment_local.set_index([subject_col, num_col])
    df = df.loc[df.index.intersection(keys)].reset_index()

    # Collapse merged forms into the canonical identity for grouping.
    if rules:
        df["Subject"] = df["Subject_Canonical"]
    if merge_z:
        df["Course_Number"] = df["Course_Number_Norm"]

    # Year range and summer.
    df = df[(df["academic_year"] >= year_start) &
            (df["academic_year"] <= year_end)]
    if exclude_summer:
        df = df[df["quarter"] != "Summer"]

    # Format / first-section filter.
    if format_mode == "Lectures only":
        df = df[df["Format_Norm"] == "Lecture"]
    elif format_mode == "First section only":
        df = df[df["section_order"] == 0]
    # else "All sections" — no filter

    if df.empty:
        return df.assign(series_key=pd.Series(dtype=str),
                         label=pd.Series(dtype=str))

    # Aggregation. Sum: collapse to one row per (Subject, Course_Number, Term_Code).
    # Sections separately: keep CRN + Format_Norm as part of the series key.
    if aggregate == "Sum sections":
        agg = (df.groupby(
                ["Subject", "Course_Number", "Term_Code",
                 "academic_year", "quarter", "calendar_year", "axis_x"],
                observed=True, as_index=False)
              .agg(Enrollment=("Enrollment", "sum"),
                   Course_Name=("Course_Name", "first"),
                   n_sections=("CRN", "nunique"),
                   Format_Norm=("Format_Norm",
                                lambda s: ", ".join(sorted(set(s))))))
        agg["series_key"] = agg["Subject"].astype(str) + " " + agg["Course_Number"].astype(str)
        agg["label"] = agg["series_key"]
    else:  # "Show sections separately"
        # CRNs are per-term, so we can't use them as a persistent series ID.
        # Group within (Term, Subject, Course_Number, Format_Norm) and number
        # the surviving sections 1, 2, 3, ... in their original order. That
        # gives "1st Lecture", "2nd Lecture" identities that persist across
        # terms — an imperfect but useful approximation since UO doesn't
        # publish stable section identifiers.
        df = df.sort_values(["Term_Code", "Subject", "Course_Number",
                             "Format_Norm", "section_order"])
        df["section_idx"] = (
            df.groupby(["Term_Code", "Subject", "Course_Number", "Format_Norm"],
                       observed=True).cumcount() + 1
        )
        agg = df.copy()
        agg["n_sections"] = 1
        agg["series_key"] = (
            agg["Subject"].astype(str) + " " + agg["Course_Number"].astype(str)
            + " · " + agg["Format_Norm"].astype(str)
            + " #" + agg["section_idx"].astype(str)
        )
        # Drop the "#1" suffix when that (course, format) only ever has one
        # section across all terms, to keep legend labels readable.
        max_idx = (agg.groupby(["Subject", "Course_Number", "Format_Norm"],
                               observed=True)["section_idx"].transform("max"))
        def _label(row, has_multi):
            base = f"{row['Subject']} {row['Course_Number']} ({row['Format_Norm']}"
            if has_multi:
                base += f" #{row['section_idx']}"
            return base + ")"
        agg["label"] = [
            _label(row, has_multi) for row, has_multi
            in zip(agg.to_dict("records"), max_idx > 1)
        ]

    return agg.sort_values(["series_key", "axis_x"]).reset_index(drop=True)


def annual_sum(plot_df: pd.DataFrame, exclude_summer: bool) -> pd.DataFrame:
    """Collapse term-level rows to one row per (series, academic_year), with
    `x_start` / `x_end` columns marking the horizontal extent of the step.

    The user-specified extents are:
      start = academic_year + 0.75   (calendar Oct 1, the AY boundary)
      end   = academic_year + 1.5    if summer excluded
            = academic_year + 1.75   if summer included
    """
    if plot_df.empty:
        return plot_df.assign(
            x_start=pd.Series(dtype=float),
            x_end=pd.Series(dtype=float),
            n_terms=pd.Series(dtype=int),
        )

    end_offset = 1.5 if exclude_summer else 1.75

    summed = (plot_df.groupby(
                  ["series_key", "label", "Subject", "Course_Number",
                   "academic_year"],
                  observed=True, as_index=False)
              .agg(Enrollment=("Enrollment", "sum"),
                   n_terms=("Term_Code", "nunique"),
                   Course_Name=("Course_Name", "first"),
                   Format_Norm=("Format_Norm",
                                lambda s: ", ".join(sorted(set(s))))))

    summed["x_start"] = summed["academic_year"] + 0.75
    summed["x_end"] = summed["academic_year"] + end_offset
    return summed.sort_values(["series_key", "academic_year"]).reset_index(drop=True)


def normalize_series(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Divide each series' Enrollment by its first non-zero value.

    Works for both term-level rows (sorted by axis_x) and AY-summed rows
    (sorted by academic_year): the function only relies on the existing
    row order within each series, which both upstream functions already
    set correctly."""
    if plot_df.empty:
        return plot_df
    out = plot_df.copy()
    out["Enrollment_raw"] = out["Enrollment"]
    new_vals = []
    for key, group in out.groupby("series_key", sort=False, observed=True):
        nz = group[group["Enrollment"] > 0]
        if nz.empty:
            new_vals.append(group["Enrollment"].astype(float))
            continue
        base = float(nz.iloc[0]["Enrollment"])
        new_vals.append(group["Enrollment"].astype(float) / base)
    out["Enrollment"] = pd.concat(new_vals).sort_index()
    return out


def build_plot(plot_df: pd.DataFrame, normalize: bool,
               year_start: int, year_end: int,
               exclude_summer: bool, annual_mode: bool) -> go.Figure:
    fig = go.Figure()

    if not plot_df.empty and not annual_mode:
        # Term-level: one point per (series, term).
        for label, group in plot_df.groupby("series_key", sort=False, observed=True):
            color = stable_color(label)
            symbol = stable_marker(label)
            display = group["label"].iloc[0]
            hover_extra = ""
            custom = None
            if "n_sections" in group.columns:
                hover_extra = (
                    "<br>Sections: %{customdata[0]}"
                    "<br>Format(s): %{customdata[1]}"
                )
                custom = list(zip(
                    group["n_sections"].fillna(1).astype(int),
                    group["Format_Norm"].astype(str),
                ))
            y_hover = "Enrollment: %{y:.2f}" if normalize else "Enrollment: %{y}"
            fig.add_trace(go.Scatter(
                x=group["axis_x"],
                y=group["Enrollment"],
                mode="lines+markers",
                name=display,
                line=dict(color=color, width=2),
                marker=dict(size=7, symbol=symbol,
                            line=dict(color=color, width=1.5)),
                customdata=custom,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "%{text}<br>"
                    f"{y_hover}"
                    f"{hover_extra}"
                    "<extra></extra>"
                ),
                text=[f"{q} {cy}" for q, cy in zip(group["quarter"],
                                                   group["calendar_year"])],
            ))

    elif not plot_df.empty and annual_mode:
        # AY-summed: one horizontal segment per (series, AY). Consecutive AYs
        # are connected by a short link from (end_n, val_n) to (start_{n+1},
        # val_{n+1}) so the eye can follow each course as a single curve.
        # If a year is missing entirely for a series (course not offered),
        # we insert a NaN break so the gap is visible.
        y_hover = "Annual total: %{y:.2f}" if normalize else "Annual total: %{y}"
        for label, group in plot_df.groupby("series_key", sort=False, observed=True):
            color = stable_color(label)
            symbol = stable_marker(label)
            display = group["label"].iloc[0]
            xs, ys, hover_text, custom = [], [], [], []
            prev_ay = None
            for _, row in group.iterrows():
                ay = int(row["academic_year"])
                if prev_ay is not None and ay - prev_ay > 1:
                    # Insert a real gap for missing years.
                    xs.append(None); ys.append(None)
                    hover_text.append(""); custom.append((0, ""))
                xs.extend([row["x_start"], row["x_end"]])
                ys.extend([row["Enrollment"], row["Enrollment"]])
                ay_label = f"AY {ay}-{(ay + 1) % 100:02d}"
                hover_text.extend([ay_label, ay_label])
                custom.extend([
                    (int(row["n_terms"]), row["Format_Norm"]),
                    (int(row["n_terms"]), row["Format_Norm"]),
                ])
                prev_ay = ay
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=display,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, symbol=symbol, color=color,
                            line=dict(color=color, width=1.5)),
                customdata=custom,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "%{text}<br>"
                    f"{y_hover}"
                    "<br>Terms summed: %{customdata[0]}"
                    "<br>Format(s): %{customdata[1]}"
                    "<extra></extra>"
                ),
                text=hover_text,
                connectgaps=False,
            ))

    # Dashed vertical lines at academic-year boundaries (calendar Oct 1).
    # The user-defined boundary is at calendar_year + 0.75.
    x_lo = year_start + 0.75 - 0.2
    x_hi = (year_end + 1) + 0.75 + 0.2
    boundary_years = range(int(x_lo), int(x_hi) + 1)
    for y in boundary_years:
        x = y + 0.75
        if x_lo <= x <= x_hi:
            fig.add_vline(x=x, line=dict(dash="dash", color="lightgray", width=1))

    # Axis and title text.
    if annual_mode:
        y_title = ("Annual total (normalized to first non-zero year)"
                   if normalize else "Annual enrollment total")
        terms_phrase = ("Fall+Winter+Spring" if exclude_summer
                        else "Fall+Winter+Spring+Summer")
        title = (f"UO course enrollment — annual totals ({terms_phrase}), "
                 f"academic years {year_start}–{year_end}")
    else:
        y_title = ("Enrollment (normalized to first non-zero term)"
                   if normalize else "Enrollment")
        note = "  ·  summers excluded" if exclude_summer else ""
        title = (f"UO course enrollment, academic years "
                 f"{year_start}–{year_end}{note}")

    fig.update_layout(
        title=title,
        xaxis_title="Academic year",
        yaxis_title=y_title,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=60, r=60, t=60, b=60),
        height=500,
        template="simple_white",
    )
    # Anchor the y-axis at zero by default. `tozero` auto-scales the upper
    # bound to fit the data but always includes 0 at the bottom. The user
    # can still pan/zoom to override interactively.
    fig.update_yaxes(rangemode="tozero")
    if normalize:
        fig.add_hline(y=1.0, line=dict(dash="dot", color="lightgray", width=1))
    return fig


def render_catalog_descriptions(catalog: pd.DataFrame,
                                selected: list[tuple[str, str]],
                                merge_z: bool = False,
                                rules: dict[str, str] | None = None) -> None:
    import re
    rules = rules or {}
    num_col = "Course_Number_Norm" if merge_z else "Course_Number"
    # The catalog uses current (canonical) subject codes, so we look up by
    # canonical subject regardless of what the user originally added.
    cat_idx = catalog.set_index(["Subject", num_col])
    for subj, num in selected:
        canon_subj = rules.get(subj, subj)
        st.markdown(f"### {canon_subj} {num}")
        lookup_num = re.sub(r"Z$", "", num) if merge_z else num
        try:
            row = cat_idx.loc[(canon_subj, lookup_num)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            title = row.get("Title", "")
            credits = row.get("Credits", "")
            html = row.get("Description_HTML") or ""
            if pd.isna(html) or not str(html).strip():
                desc = row.get("Description", "") or ""
                html = f"<p>{desc}</p>"
            header = f"**{title}** — {credits}" if title else ""
            if header:
                st.markdown(header)
            st.markdown(str(html), unsafe_allow_html=True)
            for label, col in (("Requisites", "Requisites"),
                               ("Equivalent to", "Equivalent_To"),
                               ("Additional information", "Additional_Information"),
                               ("Other notes", "Other_Notes"),
                               ("Repeatable", "Repeatable")):
                v = row.get(col)
                if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                    st.markdown(f"*{label}:* {v}")
        except KeyError:
            st.info(f"No 2025-26 catalog entry for {subj} {num}.")
        st.markdown("---")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UO Course Enrollment Dashboard",
    page_icon="📚",
    layout="wide",
)

# Guard: make sure data files are present.
if not ENROLLMENT_PARQUET.exists() or not CATALOG_CSV.exists():
    st.error(
        "Data files not found. Expected:\n\n"
        f"- `{ENROLLMENT_PARQUET}`\n"
        f"- `{CATALOG_CSV}`\n\n"
        "Run `python prepare_data.py` to build them, then commit the contents "
        "of `data/` to your repo."
    )
    st.stop()

enrollment = load_enrollment()
catalog = load_catalog()

# Session state for the selected-courses list.
if "selected" not in st.session_state:
    st.session_state.selected = []  # list of (subject, course_number) tuples

st.title("UO Course Enrollment Dashboard")
st.caption(
    "Enrollment over time for University of Oregon courses, 1990 – present. "
    "Data scraped from DuckWeb (enrollment) and the 2025-26 course catalog "
    "(descriptions)."
)

tab_plot, tab_readme = st.tabs(["Dashboard", "About / README"])

# --- Sidebar controls --------------------------------------------------------
with st.sidebar:
    merge_z = st.checkbox(
        "Combine '…Z' transferable variants",
        value=True,
        help="When on, treats courses like 'CH 221' and 'CH 221Z' as the "
             "same course (UO adds the Z suffix to mark statewide-"
             "transferable courses). Other letter suffixes (H, L, M, N) "
             "are always kept separate.",
    )

    # Subject-merge rules from modifications.csv. Loaded fresh each rerun
    # so edits to the file take effect without restarting the app.
    merge_rules = load_merge_rules()
    subject_options = build_subject_options(enrollment, merge_rules)
    display_to_canonical = dict(subject_options)

    num_col = "Course_Number_Norm" if merge_z else "Course_Number"
    subj_idx = subject_course_index(enrollment, num_col, merge_rules)

    st.header("Add a course")
    # Default to PHYS if available.
    display_labels = [d for d, _ in subject_options]
    default_idx = next(
        (i for i, (_, c) in enumerate(subject_options) if c == "PHYS"),
        0,
    )
    subj_display = st.selectbox(
        "Subject", options=display_labels, index=default_idx, key="subj_pick"
    )
    subj = display_to_canonical[subj_display]  # canonical
    nums = subj_idx.get(subj, [])
    num = st.selectbox("Course number", options=nums, key="num_pick") if nums else None

    c_add, c_clear = st.columns(2)
    with c_add:
        if st.button("➕ Add", use_container_width=True, type="primary"):
            if num is not None:
                pair = (subj, num)  # always stored as canonical
                if pair not in st.session_state.selected:
                    st.session_state.selected.append(pair)
    with c_clear:
        if st.button("🗑 Clear all", use_container_width=True):
            st.session_state.selected = []

    st.markdown("**Selected courses**")
    if not st.session_state.selected:
        st.caption("None yet — pick one above.")
    else:
        import re
        # Migrate any pre-canonicalization entries on the fly. This means
        # toggling merge_rules behavior or updating modifications.csv mid-
        # session won't strand old selections.
        migrated: list[tuple[str, str]] = []
        for s, n in st.session_state.selected:
            cs = merge_rules.get(s, s)
            migrated.append((cs, n))
        # Dedup while preserving order.
        seen: set[tuple[str, str]] = set()
        st.session_state.selected = [
            p for p in migrated if not (p in seen or seen.add(p))
        ]

        for i, (s, n) in enumerate(list(st.session_state.selected)):
            display_n = re.sub(r"Z$", "", n) if merge_z else n
            cA, cB = st.columns([0.78, 0.22])
            with cA:
                st.write(f"• {s} {display_n}")
            with cB:
                if st.button("✕", key=f"rm_{i}_{s}_{n}",
                             help="Remove this course"):
                    st.session_state.selected = [
                        p for p in st.session_state.selected
                        if p != (s, n)
                    ]
                    st.rerun()

    st.divider()
    st.header("Display options")

    yr_min = int(enrollment["academic_year"].min())
    yr_max = int(enrollment["academic_year"].max())
    year_start, year_end = st.slider(
        "Academic year range",
        min_value=yr_min, max_value=yr_max,
        value=(1990, 2025), step=1,
    )
    st.caption("Academic year 2025 = Fall 2025 through Summer 2026.")

    format_mode = st.radio(
        "Format filter",
        ("Lectures only", "First section only", "All sections"),
        index=0,
        help=("'Lectures only': sections whose Format is 'Lecture' (including "
              "primary sections with no explicit format label).  "
              "'First section only': the first row listed for each course in "
              "each term — usually the lecture.  "
              "'All sections': include every section (lab, discussion, etc.)."),
    )
    aggregate = st.radio(
        "When multiple sections survive the filter",
        ("Sum sections", "Show sections separately"),
        index=0,
    )
    exclude_summer = st.checkbox("Exclude summer term", value=True)
    annual_mode = st.checkbox(
        "Sum enrollment by academic year",
        value=False,
        help="Sum each course's enrollment across the academic year and draw "
             "a horizontal step from the start of the AY to its end "
             "(through Spring if summer is excluded; through Summer otherwise).",
    )
    normalize = st.checkbox("Normalize to first non-zero term", value=False,
                            help="Divide each series by its first non-zero value, "
                                 "so the y-axis becomes relative change. "
                                 "In annual-sum mode, normalizes to the first "
                                 "non-zero academic year instead.")

# --- Main tab ----------------------------------------------------------------
with tab_plot:
    plot_df = filter_and_aggregate(
        enrollment, st.session_state.selected,
        format_mode=format_mode, aggregate=aggregate,
        year_start=year_start, year_end=year_end,
        exclude_summer=exclude_summer,
        merge_z=merge_z,
        rules=merge_rules,
    )
    if annual_mode:
        plot_df = annual_sum(plot_df, exclude_summer=exclude_summer)
    if normalize:
        plot_df = normalize_series(plot_df)

    if not st.session_state.selected:
        st.info("Pick a subject and course in the sidebar and click **Add** "
                "to start building a plot.")
    elif plot_df.empty:
        st.warning("No matching rows for the current selection and filters. "
                   "Try widening the year range or changing the format filter.")
    else:
        fig = build_plot(plot_df, normalize=normalize,
                         year_start=year_start, year_end=year_end,
                         exclude_summer=exclude_summer,
                         annual_mode=annual_mode)
        st.plotly_chart(fig, use_container_width=True,
                        config={"toImageButtonOptions": {
                            "format": "png",
                            "filename": "uo_enrollment",
                            "scale": 2,
                        }})

        # Download buttons.
        cols = st.columns(3)
        with cols[0]:
            if annual_mode:
                csv_cols = ["Subject", "Course_Number", "Course_Name",
                            "academic_year", "x_start", "x_end",
                            "Format_Norm", "n_terms", "Enrollment"]
            else:
                csv_cols = ["Subject", "Course_Number", "Course_Name",
                            "Term_Code", "academic_year", "quarter",
                            "calendar_year", "axis_x",
                            "Format_Norm", "n_sections", "Enrollment"]
            if normalize and "Enrollment_raw" in plot_df.columns:
                csv_cols.append("Enrollment_raw")
            csv_cols = [c for c in csv_cols if c in plot_df.columns]
            csv_bytes = plot_df[csv_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download data as CSV",
                data=csv_bytes,
                file_name="uo_enrollment_selection.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with cols[1]:
            # Plotly's built-in toolbar provides PNG export. Streamlit's PNG
            # download via kaleido requires the optional kaleido package; we
            # avoid that dependency and point users to the camera icon.
            st.caption("📷 PNG export: use the camera icon in the plot's "
                       "top-right toolbar.")
        with cols[2]:
            st.caption(f"{len(plot_df):,} rows  ·  "
                       f"{plot_df['series_key'].nunique() if not plot_df.empty else 0} "
                       f"series")

        # Catalog descriptions (only when ≤ 5 courses selected).
        if 1 <= len(st.session_state.selected) <= 5:
            st.divider()
            st.subheader("2025-26 catalog descriptions")
            render_catalog_descriptions(catalog, st.session_state.selected,
                                        merge_z=merge_z, rules=merge_rules)

# --- README tab --------------------------------------------------------------
with tab_readme:
    if README_PATH.exists():
        st.markdown(README_PATH.read_text(encoding="utf-8"))
    else:
        st.warning("README.md not found.")
