import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List
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

SYSTEM_PROMPT = """
You are a football data analysis assistant for FIFA 23 players.

You have access to TOOLS that query a local machine learning system.
You MUST use these tools whenever the user asks for:
- player market value
- player overall rating
- best/most likely playing position
- comparisons between players using these metrics.

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
             "position_10": "ST"
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
           "physic": ...
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
- Use the fields in TOOL_RESULT.player such as age, potential, pace, shooting, dribbling, defending, physic.
- Explain the value difference in terms of:
  - age and remaining years at top level,
  - potential,
  - role (position_10),
  - key stats (e.g., pace + finishing for strikers).
Do NOT just restate the predicted values. Always give at least one concrete football reason based on the stats.

IMPORTANT RULES:

- You MUST NOT invent or assume any TOOL_RESULT. Only use TOOL_RESULT when it is actually provided to you in the conversation.
- You MUST NOT mention "search results", "Chinese players", or any tool output unless it comes from a real TOOL_RESULT: message.
- For any question involving specific players or their values/ratings/positions,
  you MUST:
  1) Call TOOL: search_player for each player name you need.
  2) Then call TOOL: predict_player using the short_name returned by search_player.
  3) ONLY AFTER you have TOOL_RESULT for all required players, produce FINAL_ANSWER.
- You MUST NOT answer such questions directly from your own knowledge without calling tools at least once for this query.

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

    return (
        "LLM is unreachable right now (likely network blocked). "
        "Showing offline results from local models.\n"
        f"Player: {player['long_name']} ({player['short_name']}), "
        f"age {player['age']}, club {player.get('club_name', '') or 'N/A'}, "
        f"nation {player.get('nationality_name', '') or 'N/A'}.\n"
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


def run_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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
        ]
        cols = [c for c in cols if c in df.columns]

        players = df[cols].to_dict(orient="records")
        return {"players": players}

    elif tool_name == "predict_player":
        short_name = args.get("short_name")
        if not short_name:
            raise ValueError("predict_player requires 'short_name'.")
        return predict_all_for_player(short_name)

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def agent_chat(user_query: str, max_tool_loops: int = 6) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

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
                result = run_tool(tool_name, tool_args)
                tool_payload = json.dumps(result)
                saw_tool_result = True
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
            return final

        # 4) No TOOL call and no FINAL_ANSWER:
        #    If we haven't used tools yet for this query, force it to.
        if not saw_tool_result:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not call any tool. For this query you MUST first call "
                        "search_player and then predict_player as per the TOOL instructions. "
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
