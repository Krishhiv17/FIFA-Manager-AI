import streamlit as st
from typing import List, Dict

from src.team_optimizer import (
    TeamConstraints,
    get_formation,
    load_player_pool,
    build_team_with_relaxation,
    compute_tci,
)
from src.transfer_suggester import suggest_replacements_for_player
from src.fifa_models_service import predict_all_for_player, find_players_by_name


# ----------------- Styling ----------------- #
PRIMARY = "#0B1E3A"  # dark blue
ACCENT = "#4C6FFF"
BG_DARK = "#0F172A"
CARD = "#111827"
TEXT = "#E5E7EB"

st.set_page_config(
    page_title="FIFA Manager AI",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .card {{
        background: {CARD};
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #1F2937;
        margin-bottom: 8px;
    }}
    .metric {{
        background: {CARD};
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid #1F2937;
    }}
    .accent {{
        color: {ACCENT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Helpers ----------------- #


def _fmt_val(eur: float) -> str:
    try:
        eur = float(eur)
    except Exception:
        return str(eur)
    if abs(eur) >= 1_000_000:
        return f"€{eur/1_000_000:.1f}M"
    if abs(eur) >= 1_000:
        return f"€{eur/1_000:.1f}K"
    return f"€{int(eur)}"


def _parse_team_text(raw: str) -> List[Dict[str, str]]:
    team = []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        name, pos = line.split(":", 1)
        name = name.strip()
        pos = pos.strip().upper()
        if name and pos:
            team.append({"short_name": name, "position_10": pos})
    return team


@st.cache_data
def _load_pool():
    return load_player_pool()


# ----------------- Sidebar Controls ----------------- #
st.sidebar.title("Controls")
section = st.sidebar.radio("Mode", ["Build Team", "Player Replace", "Compare Players"])

style = st.sidebar.selectbox("Style", ["balanced", "counter", "high_press", "possession"], index=0)
formation_code = st.sidebar.selectbox("Formation", ["433_cdm", "433_cam"], index=0)
budget = st.sidebar.number_input("Budget (€)", min_value=0, value=500_000_000, step=50_000_000, format="%i")
max_age = st.sidebar.number_input("Max age", min_value=0, value=30, step=1)
min_overall = st.sidebar.number_input("Min overall", min_value=0, max_value=99, value=82, step=1)
prefer_styles = st.sidebar.text_input("Preferred playstyles (comma-separated)")
bench_size = st.sidebar.slider("Bench size", min_value=0, max_value=5, value=3)
gk_reserve_pct = st.sidebar.slider("Reserve % for GK", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
mandatory_text = st.sidebar.text_input("Mandatory players (comma-separated)")
blocked_text = st.sidebar.text_input("Blocked players (comma-separated)")

pref_styles_list = [s.strip() for s in prefer_styles.split(",") if s.strip()]
mandatory_list = [s.strip() for s in mandatory_text.split(",") if s.strip()]
blocked_list = [s.strip() for s in blocked_text.split(",") if s.strip()]

constraints = TeamConstraints(
    formation_code=formation_code,
    style=style,
    budget_eur=budget if budget > 0 else None,
    max_age=max_age if max_age > 0 else None,
    min_overall=min_overall if min_overall > 0 else None,
    prefer_playstyles=pref_styles_list,
    gk_reserve_pct=gk_reserve_pct,
    mandatory_players=mandatory_list,
    blocked_players=blocked_list,
    bench_size=bench_size,
)

pool = _load_pool()

# ----------------- Main Sections ----------------- #
st.title("FIFA Manager AI")

if section == "Build Team":
    st.header("Build Team")
    if st.button("Run Optimizer", type="primary"):
        formation = get_formation(constraints.formation_code)
        try:
            team, summary = build_team_with_relaxation(pool, formation, constraints)
            tci = summary.get("tci") or compute_tci(team, constraints, bench=summary.get("bench"))
            st.subheader(f"{formation.name} | {constraints.style.title()}")
            st.write(f"Total value {_fmt_val(summary['total_value_eur'])} | Avg overall {summary['avg_overall']:.2f}")
            st.write(
                f"TCI {tci['tci']:.1f} (pos {tci['positional_fit']:.1f}, role {tci['role_playstyle_fit']:.1f}, "
                f"style {tci['style_compliance']:.1f}, line {tci['line_balance']:.1f}, depth {tci['depth']:.1f})"
            )
            cols = st.columns(3)
            for idx, item in enumerate(team):
                p = item["player"]
                col = cols[idx % 3]
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                        <strong>{item['slot']}</strong> — {item['role_hint']}<br>
                        <span class="accent">{p['short_name']}</span> ({p.get('playstyle','')})<br>
                        Overall: {p['overall']} | Value: {_fmt_val(p['value_eur'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            bench = summary.get("bench") or []
            if bench:
                st.subheader("Bench")
                cols_b = st.columns(3)
                for idx, b in enumerate(bench):
                    p = b["player"]
                    col = cols_b[idx % 3]
                    with col:
                        st.markdown(
                            f"""
                            <div class="card">
                            <strong>{b['slot']}</strong> ({p.get('position_10','')})<br>
                            <span class="accent">{p['short_name']}</span> ({p.get('playstyle','')})<br>
                            Overall: {p['overall']} | Value: {_fmt_val(p['value_eur'])}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            relax = summary.get("relaxations") or []
            if relax:
                st.info("Relaxations applied: " + "; ".join(relax))
        except Exception as e:
            st.error(f"Failed to build team: {e}")

elif section == "Player Replace":
    st.header("Replace a Player")
    with st.form("replace_form"):
        current_name = st.text_input("Player to replace (short_name)")
        current_pos = st.text_input("Position (optional, e.g., ST, CM, LW)").upper().strip()
        desired_ps = st.text_input("Desired playstyles (optional, comma-separated)")
        desired_list = [s.strip() for s in desired_ps.split(",") if s.strip()]
        submitted = st.form_submit_button("Suggest Replacements", type="primary")
    if submitted:
        if not current_name:
            st.warning("Please provide a player name.")
        else:
            curr = {"short_name": current_name, "position_10": current_pos}
            try:
                suggestions = suggest_replacements_for_player(
                    current_player=curr,
                    constraints=constraints,
                    desired_playstyles=desired_list,
                    max_suggestions=3,
                )
                if not suggestions:
                    st.warning("No suggestions under current constraints.")
                else:
                    for s in suggestions:
                        c = s["candidate"]
                        st.markdown(
                            f"""
                            <div class="card">
                            <strong>{c['short_name']}</strong> — {c.get('playstyle','')} ({c.get('position_10','')})<br>
                            Overall: {c['overall']} | Value: {_fmt_val(c['value_eur'])}<br>
                            Pace: {c.get('pace','?')} | Physic: {c.get('physic','?')}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Error suggesting replacements: {e}")

elif section == "Compare Players":
    st.header("Compare Players")
    col1, col2 = st.columns(2)
    with col1:
        name_a = st.text_input("Player A (short_name)", value="T. Courtois")
    with col2:
        name_b = st.text_input("Player B (short_name)", value="De Gea")
    if st.button("Compare", type="primary"):
        try:
            res = []
            for nm in [name_a, name_b]:
                row = find_players_by_name(nm, top_k=1)
                if row.empty:
                    st.warning(f"Player not found: {nm}")
                    continue
                pa = predict_all_for_player(row.iloc[0]['short_name'])
                res.append(pa)
            if len(res) == 2:
                colA, colB = st.columns(2)
                for col, pa in zip([colA, colB], res):
                    player = pa["player"]; actual = pa["actual"]; preds = pa["predictions"]
                    with col:
                        st.markdown(
                            f"""
                            <div class="card">
                            <strong>{player['long_name']} ({player['short_name']})</strong><br>
                            Dataset: {_fmt_val(actual['value_eur'])}, overall {actual['overall']}, pos {actual['position_10']}<br>
                            Pred: {_fmt_val(preds['value']['value_pred'])}, overall {preds['overall']['overall_pred']:.1f}, pos {preds['position']['position_pred']}<br>
                            Playstyle: {player.get('playstyle','Unspecified')} | Age: {player['age']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Comparison failed: {e}")
