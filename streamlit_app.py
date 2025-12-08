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


def _player_names(pool) -> List[str]:
    return sorted(pool["short_name"].dropna().unique().tolist())


def _playstyle_options(pool) -> List[str]:
    return sorted([p for p in pool["playstyle"].dropna().unique().tolist() if p])


# ----------------- Sidebar Controls ----------------- #
st.sidebar.title("Controls")
section = st.sidebar.radio("Mode", ["Build Team", "Player Replace", "Compare Players", "Player Info"])

pool = _load_pool()
all_player_names = _player_names(pool)
all_playstyles = _playstyle_options(pool)

style = st.sidebar.selectbox("Style", ["balanced", "counter", "high_press", "possession"], index=0)
formation_code = st.sidebar.selectbox("Formation", ["433_cdm", "433_cam"], index=0)
budget = st.sidebar.number_input("Budget (€)", min_value=0, value=500_000_000, step=50_000_000, format="%i")
max_age = st.sidebar.number_input("Max age", min_value=0, value=30, step=1)
min_overall = st.sidebar.number_input("Min overall", min_value=0, max_value=99, value=82, step=1)
prefer_styles_sel = st.sidebar.multiselect(
    "Preferred player playstyles",
    options=all_playstyles,
    default=[],
)
bench_size = st.sidebar.slider("Bench size", min_value=0, max_value=5, value=3)
gk_reserve_pct = st.sidebar.slider("Reserve % for GK", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
mandatory_list = st.sidebar.multiselect("Mandatory players", options=all_player_names, default=[])
blocked_list = st.sidebar.multiselect("Blocked players", options=all_player_names, default=[])
pref_styles_list = list(prefer_styles_sel)

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
        current_name = st.selectbox("Player to replace", all_player_names)
        current_pos = st.text_input("Position (optional, e.g., ST, CM, LW)").upper().strip()
        desired_list = st.multiselect("Desired playstyles (optional)", options=all_playstyles)
        submitted = st.form_submit_button("Suggest Replacements", type="primary")
    if submitted:
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
        name_a = st.selectbox("Player A", all_player_names, index=all_player_names.index("T. Courtois") if "T. Courtois" in all_player_names else 0)
    with col2:
        name_b = st.selectbox("Player B", all_player_names, index=all_player_names.index("De Gea") if "De Gea" in all_player_names else 0)
    if st.button("Compare", type="primary"):
        try:
            res = []
            for nm in [name_a, name_b]:
                pa = predict_all_for_player(nm)
                res.append(pa)
            if len(res) == 2:
                def describe(pa):
                    player = pa["player"]
                    preds = pa["predictions"]
                    top3 = preds["position"].get("position_top3", [])
                    top3_str = ", ".join(f"{p['position']} {p['prob']*100:.0f}%" for p in top3)
                    stats = {k: player.get(k) for k in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]}
                    strengths = sorted([k for k,v in stats.items() if isinstance(v,(int,float))], key=lambda k: stats[k], reverse=True)[:2]
                    strength_txt = ", ".join(f"{s} {stats[s]}" for s in strengths)
                    return (
                        f"{player['long_name']} ({player['short_name']}), age {player['age']}, playstyle {player.get('playstyle','Unspecified')}. "
                        f"Predicted: value {_fmt_val(preds['value']['value_pred'])}, overall {preds['overall']['overall_pred']:.1f}, "
                        f"position {preds['position']['position_pred']} (top3: {top3_str}). "
                        f"Key strengths: {strength_txt}."
                    ), preds, player

                desc_a, preds_a, player_a = describe(res[0])
                desc_b, preds_b, player_b = describe(res[1])
                st.write("\n\n".join([desc_a, desc_b]))

                # Explain winner with a looser narrative
                val_a = preds_a["value"]["value_pred"]
                val_b = preds_b["value"]["value_pred"]
                ov_a = preds_a["overall"]["overall_pred"]
                ov_b = preds_b["overall"]["overall_pred"]
                age_a = player_a.get("age")
                age_b = player_b.get("age")
                strengths_a = [k for k,v in res[0]["player"].items() if k in ["pace","shooting","passing","dribbling","defending","physic"] and isinstance(v,(int,float))]
                strengths_b = [k for k,v in res[1]["player"].items() if k in ["pace","shooting","passing","dribbling","defending","physic"] and isinstance(v,(int,float))]

                hi_val = player_a if val_a >= val_b else player_b
                lo_val = player_b if val_a >= val_b else player_a
                hi_val_pred = max(val_a, val_b)
                lo_val_pred = min(val_a, val_b)

                hi_ov = player_a if ov_a >= ov_b else player_b
                lo_ov = player_b if ov_a >= ov_b else player_a
                hi_ov_pred = max(ov_a, ov_b)
                lo_ov_pred = min(ov_a, ov_b)

                reason_lines = []
                reason_lines.append(
                    f"{hi_val['short_name']} carries the bigger price tag ({_fmt_val(hi_val_pred)} vs {_fmt_val(lo_val_pred)}) "
                    f"because their projected overall is higher ({hi_ov_pred:.1f} vs {lo_ov_pred:.1f}) and they rate better in core attributes "
                    f"like {', '.join([s for s in (strengths_a if hi_val is player_a else strengths_b)[:2]])}."
                )
                if age_a and age_b and age_a != age_b:
                    younger = player_a if age_a < age_b else player_b
                    reason_lines.append(f"{younger['short_name']} is younger, which props up long-term value even if current overall lags.")
                if preds_a["position"]["position_pred"] != preds_b["position"]["position_pred"]:
                    reason_lines.append(
                        f"Role fit differs: {player_a['short_name']} profiles as {preds_a['position']['position_pred']}, "
                        f"while {player_b['short_name']} is best as {preds_b['position']['position_pred']}."
                    )
                st.info(" ".join(reason_lines))
        except Exception as e:
            st.error(f"Comparison failed: {e}")

elif section == "Player Info":
    st.header("Player Info")
    player_name = st.selectbox("Player", all_player_names, index=0)
    if st.button("Show Info", type="primary"):
        try:
            pa = predict_all_for_player(player_name)
            player = pa["player"]
            preds = pa["predictions"]
            top3 = preds["position"].get("position_top3", [])
            top3_str = ", ".join(f"{p['position']} {p['prob']*100:.0f}%" for p in top3)
            stats = {k: player.get(k) for k in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]}
            strengths = sorted([k for k,v in stats.items() if isinstance(v,(int,float))], key=lambda k: stats[k], reverse=True)[:3]
            strength_txt = ", ".join(f"{s} {stats[s]}" for s in strengths)
            why = "Value/overall are driven by " + strength_txt
            st.markdown(
                f"""
                <div class="card">
                <strong>{player['long_name']} ({player['short_name']})</strong><br>
                Age {player.get('age','?')}, Playstyle {player.get('playstyle','Unspecified')}<br>
                Predicted value: {_fmt_val(preds['value']['value_pred'])} | Predicted overall: {preds['overall']['overall_pred']:.1f}<br>
                Position: {preds['position']['position_pred']} (top3: {top3_str})<br>
                {why}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Lookup failed: {e}")
