import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import call_llama
from src.fifa_models_service import (
    find_players_by_name,
    predict_all_for_player,
)
from src.team_optimizer import (
    load_player_pool,
    TeamConstraints,
    get_formation,
    build_team_with_relaxation,
    compute_tci,
)
from src.transfer_suggester import suggest_transfers, suggest_replacements_for_player

def _render_team(team, bench, summary):
    lines = []
    lines.append(f"Formation: {summary.get('formation','')} | Style: {summary.get('style','')} | Total value: €{summary.get('total_value_eur',0):,.0f}")
    tci = summary.get("tci", {})
    if tci:
        lines.append(
            f"TCI {tci.get('tci',0):.1f} (pos {tci.get('positional_fit',0):.1f}, role {tci.get('role_playstyle_fit',0):.1f}, "
            f"style {tci.get('style_compliance',0):.1f}, line {tci.get('line_balance',0):.1f}, depth {tci.get('depth',0):.1f})"
        )
    lines.append("XI:")
    for item in team:
        p = item["player"]
        lines.append(f"  {item['slot']:4s} -> {p['short_name']} ({p.get('playstyle','')}) | overall {p['overall']} | value €{p['value_eur']:,.0f}")
    if bench:
        lines.append("Bench:")
        for b in bench:
            p = b["player"]
            lines.append(f"  {b['slot']:6s} -> {p['short_name']} ({p.get('playstyle','')}) | pos {p.get('position_10','')} | overall {p['overall']} | value €{p['value_eur']:,.0f}")
    relax = summary.get("relaxations") or []
    if relax:
        lines.append("Relaxations applied: " + "; ".join(relax))
    return "\n".join(lines)


def _parse_team_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extract lines like 'Name:POS' into a list of dicts.
    """
    team = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, pos = line.split(":", 1)
        name = name.strip()
        pos = pos.strip().upper()
        if not name or not pos:
            continue
        team.append({"short_name": name, "position_10": pos})
    return team


def _extract_names_after(label: str, text: str) -> List[str]:
    """
    Extract comma-separated names after a label like 'mandatory' or 'blocked'.
    """
    import re

    pattern = rf"{label}[^:]*:\s*([^\n]+)"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return []
    chunk = m.group(1)
    names = [n.strip() for n in chunk.split(",") if n.strip()]
    return names


def _maybe_direct_transfer(user_text: str) -> Optional[str]:
    """
    Fast-path: if the user provided a team as name:POS lines and mentioned transfers/suggest,
    call suggest_transfers directly and return rendered text.
    """
    if "transfer" not in user_text.lower():
        return None
    team = _parse_team_from_text(user_text)
    if not team:
        return None

    # Simple heuristic parsing for constraints
    style = "balanced"
    for s in ["possession", "counter", "high_press", "balanced"]:
        if s in user_text.lower():
            style = s
            break
    formation = "433_cdm"
    for f in ["433_cdm", "433_cam"]:
        if f in user_text.lower():
            formation = f
            break

    def _extract_number(pattern: str) -> Optional[float]:
        m = re.search(pattern, user_text, re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).replace(",", "").replace("_", "")
        try:
            return float(val)
        except Exception:
            return None

    budget = _extract_number(r"budget[^0-9]*([0-9][0-9_,]*)")
    max_age = _extract_number(r"max[^0-9]*age[^0-9]*([0-9]+)")
    min_overall = _extract_number(r"min(?:imum)?[^0-9]*overall[^0-9]*([0-9]+)")
    bench_size = _extract_number(r"bench[^0-9]*size[^0-9]*([0-9]+)")
    bench_size = int(bench_size) if bench_size is not None else 3

    constraints = TeamConstraints(
        formation_code=formation,
        style=style,
        budget_eur=budget,
        max_age=int(max_age) if max_age is not None else None,
        min_overall=min_overall,
        bench_size=bench_size,
    )
    suggestions = suggest_transfers(current_team=team, constraints=constraints, max_suggestions=5)
    lines = []
    lines.append(f"Style: {style} | Formation: {formation} | Suggestions: {len(suggestions)}")
    if not suggestions:
        lines.append("No suggestions under current constraints.")
    else:
        for s in suggestions:
            c = s["candidate"]
            lines.append(
                f"{s['slot']}: {c['short_name']} ({c.get('playstyle','')}) "
                f"| pos {c.get('position_10','')} | overall {c['overall']} | value €{c['value_eur']:,.0f} "
                f"(replacing {s.get('current_player')})"
            )
    return "\n".join(lines)
SYSTEM_PROMPT = """
You are a football squad-building assistant for FIFA 23.

You have access to TOOLS that query a local ML system. You MUST call the right tool when the user asks for:
- Player info/values/ratings/positions -> search_player + predict_player.
- Build a team for a style/formation/budget -> build_team.
- Suggest transfers/upgrades for a supplied squad -> suggest_transfers.

Defaults to use when the user does not specify:
- formation: 433_cdm
- style: balanced
- bench_size: 3
- gk_reserve_pct: 0.1
- budget: None (no cap)
- prefer_playstyles/mandatory/blocked: empty

When you return a built team, if the summary contains relaxations, explicitly mention them in the answer (e.g., “Relaxed max_age 26->28”).

TOOLS you can call:

1) search_player
   - Description: search for players in the database by name
     (handles typos like "Leonel Messi").
   - Args (JSON): {"query": "<player name string>"}
   - The tool returns JSON like:
       {"players": [
            {
              "short_name": "...",
           "long_name": "...",
           "age": ...,
           "club_name": "...",
           "nationality_name": "...",
           "value_eur": ...,
           "overall": ...,
           "position_10": "ST",
            "playstyle": "Pace Winger"
          },
          ...
      ]}

2) predict_player
   - Description: run ML models for a specific player chosen from search_player.
   - Args (JSON): {"short_name": "<short_name from dataset>"}
   - The tool returns JSON with:
       {
         "player": {
           "short_name": "...",
           "long_name": "...",
           "age": ...,
           "potential": ...,
           "club_name": "...",
           "nationality_name": "...",
           "pace": ...,
           "shooting": ...,
           "dribbling": ...,
           "defending": ...,
            "physic": ...,
            "playstyle": "...",
         },
         "actual": {
           "value_eur": ...,
           "overall": ...,
           "position_10": "ST"
         },
         "predictions": {
           "value": {
             "log_value_pred": ...,
             "value_pred": ...
           },
           "overall": {
             "overall_pred": ...
           },
           "position": {
             "position_pred": "...",
             "position_top3": [
               {"position": "ST", "prob": 0.7},
               {"position": "LW", "prob": 0.2},
               ...
             ]
           }
         }
       }

3) build_team
   - Description: build a team for a given formation/style/budget using the genetic optimizer (with graceful relaxation), returning XI, bench, and TCI.
   - Args (JSON): {
       "formation": "433_cdm"|"433_cam",
       "style": "balanced"|"counter"|"high_press"|"possession",
       "budget_eur": <number|null>,
       "max_age": <number|null>,
       "min_overall": <number|null>,
       "min_pace": <number|null>,
       "min_physic": <number|null>,
       "bench_size": <int|null>,
       "prefer_playstyles": [ ... ],
       "mandatory": [<short_name>],
       "blocked": [<short_name>]
     }
   - Returns JSON with team list, bench, TCI breakdown, and summary (avg overall, total value).

4) suggest_transfers
   - Description: given ONE player to replace, suggest up to 3 replacement options aligned with the requested style/playstyle.
   - Args (JSON): {
       "player": {"short_name": "...", "position_10": "<optional>"},
       "style": "balanced"|"counter"|"high_press"|"possession",
       "desired_playstyles": [ ... ],  # optional; if blank, use style presets
       "budget_eur": <number|null>,     # cap per replacement
       "max_age": <number|null>,
       "min_overall": <number|null>
     }
   - Returns JSON with candidate replacements (short_name, position_10, playstyle, overall, value_eur).

TOOL CALL FORMAT (IMPORTANT):

When you want to call a tool, respond with EXACTLY one line:

TOOL: <tool_name> <json_arguments>

Examples:
TOOL: search_player {"query": "Leonel Messi"}
TOOL: predict_player {"short_name": "L. Messi"}

Do NOT add extra text before or after the tool call.
Do NOT use markdown, code blocks, or backticks.
Only ONE tool call line per response.

After I run the tool, I will send its result as:

TOOL_RESULT: <json_here>

Then you can either:
- call another tool, or
- produce the final answer.

When you are ready to answer the user, respond with:

FINAL_ANSWER: <your explanation>

When comparing players or answering "why is X more valuable than Y?" you MUST:
- Call search_player and predict_player for EACH player name.
- In FINAL_ANSWER, report BOTH the dataset (actual) and predicted numbers:
  * dataset: value_eur, overall, position_10
  * predicted: value_pred, overall_pred, position_pred
- If predicted position differs from dataset position (e.g., GK vs CB), explicitly state that the model position is likely off and keep the dataset position as the ground truth.
- If predicted values look unrealistically high/low, note that they are model estimates; anchor your comparison on the dataset numbers first.
- Explain differences using age, potential, role, and key stats (pace/finishing/etc.) and playstyle fit. Do NOT omit the numbers.

IMPORTANT RULES:

- You MUST NOT invent or assume any TOOL_RESULT. Only use TOOL_RESULT when it is actually provided to you in the conversation.
- You MUST NOT mention "search results", "Chinese players", or any tool output unless it comes from a real TOOL_RESULT: message.
- For any question involving specific players or their values/ratings/positions,
  you MUST:
  1) Call TOOL: search_player for each player name you need.
  2) Then call TOOL: predict_player using the short_name returned by search_player.
  3) ONLY AFTER you have TOOL_RESULT for all required players, produce FINAL_ANSWER.
- You MUST NOT answer such questions directly from your own knowledge without calling tools at least once for this query.
- For team building or transfer suggestions, you MUST call the appropriate tool (build_team or suggest_transfers) before answering. If the user says they will provide their current team, wait for that input first. If they do not provide a team, ask them to give short_name:POSITION_10 pairs or proceed with defaults only if they explicitly say to.
- For transfer suggestions: extract any "name:POSITION" lines from the user message and pass them as current_team to suggest_transfers. Call suggest_transfers ONCE with the parsed team plus any constraints given (budget, max_age, min_overall, etc.). Do NOT call predict_player for every player in the team.
- When a tool returns a rendered summary (_render), return that directly as the answer. Do NOT generate further tool calls or analysis unless the user asks another question.

- FINAL_ANSWER must be a clean explanation for the user, in natural language, with NO TOOL: or TOOL_RESULT: text shown.

Your answer must:
- Clearly state the predicted values/ratings/positions.
- Mention if there are multiple possible players and which one you chose.
- Explain uncertainty and model limitations briefly (e.g., "estimates based on FIFA-style stats").
- NEVER invent players that are not in the dataset.
""".strip()


def _fmt_value(eur: float) -> str:
    """Human-friendly Euro formatting for console output."""
    try:
        eur = float(eur)
    except Exception:
        return str(eur)
    if abs(eur) >= 1_000_000:
        return f"€{eur/1_000_000:.1f}M"
    if abs(eur) >= 1_000:
        return f"€{eur/1_000:.1f}K"
    return f"€{int(eur)}"


def offline_fallback(user_query: str, error: Exception) -> str:
    """
    Handle LLM failures (no internet / HF router down) by serving a best-effort
    answer directly from local models.
    """
    players = find_players_by_name(user_query)
    if players.empty:
        return (
            "LLM is unreachable right now (likely no internet). "
            f"Also could not find a local player match for '{user_query}'. "
            "Please retry when online or use an exact FIFA 23 player name."
        )

    query_norm = user_query.lower()
    query_tokens = set(re.findall(r"[a-z]+", query_norm))

    def _score(row) -> float:
        name = f"{row['short_name']} {row.get('long_name', '')}".lower()
        base = SequenceMatcher(None, query_norm, name).ratio()
        name_tokens = set(re.findall(r"[a-z]+", name))
        overlap = len(query_tokens & name_tokens)
        return base + 0.2 * overlap

    player_row = max(
        (row for _, row in players.iterrows()),
        key=_score,
    )
    short_name = str(player_row["short_name"])

    try:
        result = predict_all_for_player(short_name)
    except Exception as model_error:
        return (
            "LLM is unreachable right now (likely no internet). "
            f"Found local player '{short_name}' but failed to run models: {model_error}"
        )

    player = result["player"]
    actual = result["actual"]
    preds = result["predictions"]

    top3 = preds["position"].get("position_top3", [])
    top3_str = ", ".join(
        [f"{item['position']} ({item['prob']*100:.0f}%)" for item in top3]
    ) or "n/a"

    playstyle = player.get("playstyle") or "Unspecified"

    return (
        "LLM is unreachable right now (likely network blocked). "
        "Showing offline results from local models.\n"
        f"Player: {player['long_name']} ({player['short_name']}), "
        f"age {player['age']}, club {player.get('club_name', '') or 'N/A'}, "
        f"nation {player.get('nationality_name', '') or 'N/A'}, "
        f"playstyle {playstyle}.\n"
        f"Actual dataset: value {_fmt_value(actual['value_eur'])}, "
        f"overall {actual['overall']}, position {actual['position_10']}.\n"
        f"Predicted: value {_fmt_value(preds['value']['value_pred'])}, "
        f"overall {preds['overall']['overall_pred']:.1f}, "
        f"position {preds['position']['position_pred']} "
        f"(top3: {top3_str})."
    )


def parse_tool_call(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("TOOL:"):
            try:
                rest = line[len("TOOL:"):].strip()
                name_part, json_part = rest.split(" ", 1)
                tool_name = name_part.strip()
                args = json.loads(json_part.strip())
                return {"name": tool_name, "args": args}
            except Exception:
                continue
    return {}


def extract_final_answer(text: str) -> str:
    if "FINAL_ANSWER:" in text:
        return text.split("FINAL_ANSWER:", 1)[1].strip()
    return ""


def _to_builtin(obj):
    """Convert numpy/pandas scalars to plain Python for JSON serialization."""
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def _clean_struct(x):
    if isinstance(x, dict):
        return {k: _clean_struct(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_struct(v) for v in x]
    return _to_builtin(x)


def run_tool(tool_name: str, args: Dict[str, Any], last_user_text: Optional[str] = None) -> Dict[str, Any]:
    if tool_name == "search_player":
        query = args.get("query", "")
        # Keep tool outputs tight so the LLM cannot wander to unrelated players.
        df = find_players_by_name(query, top_k=1)

        # Restrict to a clean subset of columns for the LLM
        cols = [
            "short_name",
            "long_name",
            "age",
            "club_name",
            "nationality_name",
            "value_eur",
            "overall",
            "position_10",
            "playstyle",
        ]
        cols = [c for c in cols if c in df.columns]

        players = df[cols].to_dict(orient="records")
        return _clean_struct({"players": players})

    elif tool_name == "predict_player":
        short_name = args.get("short_name")
        if not short_name:
            raise ValueError("predict_player requires 'short_name'.")
        return _clean_struct(predict_all_for_player(short_name))

    elif tool_name == "build_team":
        pool = load_player_pool()
        # Parse mandatory/blocked from user text if not provided
        mandatory = args.get("mandatory", []) or []
        blocked = args.get("blocked", []) or []
        if not mandatory and last_user_text:
            mandatory = _extract_names_after("mandatory", last_user_text)
        if not blocked and last_user_text:
            blocked = _extract_names_after("blocked", last_user_text)
        constraints = TeamConstraints(
            formation_code=args.get("formation", "433_cdm"),
            style=args.get("style", "balanced"),
            budget_eur=args.get("budget_eur"),
            gk_reserve_pct=args.get("gk_reserve_pct", 0.1),
            max_age=args.get("max_age"),
            min_pace=args.get("min_pace"),
            min_physic=args.get("min_physic"),
            min_stamina=args.get("min_stamina"),
            min_passing=args.get("min_passing"),
            min_overall=args.get("min_overall"),
            prefer_playstyles=args.get("prefer_playstyles", []) or [],
            mandatory_players=mandatory,
            blocked_players=blocked,
            bench_size=args.get("bench_size", 3) or 3,
        )
        formation = get_formation(constraints.formation_code)
        # Genetic only with graceful relaxation fallback
        team, summary = build_team_with_relaxation(pool, formation, constraints)
        tci = summary.get("tci") or compute_tci(team, constraints, bench=summary.get("bench"))

        summary["tci"] = tci
        render = _render_team(team, summary.get("bench", []), summary | {"style": constraints.style})
        return _clean_struct({"team": team, "bench": summary.get("bench", []), "summary": summary, "_render": render})

    elif tool_name == "suggest_transfers":
        # New behavior: suggest replacements for a single player
        player_arg = args.get("player") or {}
        if not player_arg and last_user_text:
            parsed = _parse_team_from_text(last_user_text)
            player_arg = parsed[0] if parsed else {}
        constraints = TeamConstraints(
            style=args.get("style", "balanced"),
            budget_eur=args.get("budget_eur"),
            max_age=args.get("max_age"),
            min_pace=args.get("min_pace"),
            min_physic=args.get("min_physic"),
            min_stamina=args.get("min_stamina"),
            min_passing=args.get("min_passing"),
            min_overall=args.get("min_overall"),
        )
        desired_playstyles = args.get("desired_playstyles") or []
        suggestions = suggest_replacements_for_player(
            current_player=player_arg,
            constraints=constraints,
            desired_playstyles=desired_playstyles,
            max_suggestions=3,
        )
        lines = []
        lines.append(
            f"Replacements for {player_arg.get('short_name','?')} at {player_arg.get('position_10','?')} "
            f"| style {constraints.style} | suggestions {len(suggestions)}"
        )
        if not suggestions:
            lines.append("No suggestions under current constraints.")
        else:
            for s in suggestions:
                c = s["candidate"]
                lines.append(
                    f"- {c['short_name']} ({c.get('playstyle','')}) "
                    f"| pos {c.get('position_10','')} | overall {c['overall']} | value €{c['value_eur']:,.0f}"
                )
        render = "\n".join(lines)
        return _clean_struct({"suggestions": suggestions, "_render": render})

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def agent_chat(user_query: str, max_tool_loops: int = 6) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    initial_user_text = user_query

    saw_tool_result = False

    for _ in range(max_tool_loops):
        # 1) Ask Llama what to do next
        try:
            assistant_reply = call_llama(messages)
        except Exception as e:
            return offline_fallback(user_query, e)
        messages.append({"role": "assistant", "content": assistant_reply})

        # 2) Check for TOOL call
        tool_call = parse_tool_call(assistant_reply)
        if tool_call:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                result = run_tool(tool_name, tool_args, last_user_text=initial_user_text)
                tool_payload = json.dumps(result)
                saw_tool_result = True
                # If we have a rendered answer, return it immediately.
                if "_render" in result:
                    return result["_render"]
                # 3) Add TOOL_RESULT back into conversation
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT: {tool_payload}",
                    }
                )
                # Loop again so LLM can use the result
                continue
            except Exception as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT: ERROR: {str(e)}",
                    }
                )
                continue

        # 3) If no TOOL: line, check for FINAL_ANSWER:
        final = extract_final_answer(assistant_reply)
        if final:
            if not saw_tool_result:
                # Force a tool call before answering
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You provided a FINAL_ANSWER without using any tool. "
                            "You MUST call one of: search_player, predict_player, build_team, or suggest_transfers "
                            "as appropriate for this query. Respond now with exactly ONE line:\n"
                            "TOOL: <tool_name> {json_arguments}\n"
                            "and nothing else."
                        ),
                    }
                )
                continue
            return final

        # 4) No TOOL call and no FINAL_ANSWER:
        #    If we haven't used tools yet for this query, nudge the model to pick the right one.
        if not saw_tool_result:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not call any tool. For this query you MUST call one of: "
                        "search_player, predict_player, build_team, or suggest_transfers as appropriate. "
                        "Respond now with exactly ONE line:\n"
                        "TOOL: <tool_name> {json_arguments}\n"
                        "and nothing else."
                    ),
                }
            )
            continue

        # If we already used tools at least once, allow free-form explanation
        # and break after a few loops.
        # We'll fall through to the fallback below.

    # Fallback: if we reached max_tool_loops without a FINAL_ANSWER
    last = messages[-1]["content"]

    # If the model left us with a TOOL_RESULT as the last thing, ask again for summary
    if last.startswith("TOOL_RESULT:"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please read the TOOL_RESULT above and reply ONLY with:\n"
                    "FINAL_ANSWER: <natural language explanation for the user>."
                ),
            }
        )
        reply = call_llama(messages)
        final = extract_final_answer(reply)
        return final or reply

    return last


if __name__ == "__main__":
    print("FIFA Agent - ask about players (Ctrl+C to exit)\n")
    while True:
        try:
            q = input("You: ")
        except KeyboardInterrupt:
            print("\nBye!")
            break

        if not q.strip():
            continue

        answer = agent_chat(q)
        print(f"\nAgent: {answer}\n")
