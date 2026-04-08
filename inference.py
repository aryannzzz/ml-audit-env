#!/usr/bin/env python3
"""
Competition-safe inference script for ML Audit Bench.

Stdout emits only mandatory protocol lines:
- [START] once per episode
- [STEP] once per successful /step call
- [END] once per episode (always, even on exceptions)
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception as exc:  # pragma: no cover - defensive for validator runtime variance
    requests = None
    REQUESTS_IMPORT_ERROR = exc
else:
    REQUESTS_IMPORT_ERROR = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - defensive for validator runtime variance
    OpenAI = None
    OPENAI_IMPORT_ERROR = exc
else:
    OPENAI_IMPORT_ERROR = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid integer for {name}: {raw!r}. Using {default}.", file=sys.stderr, flush=True)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid float for {name}: {raw!r}. Using {default}.", file=sys.stderr, flush=True)
        return default


# Mandatory defaults per competition specification
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"

BENCHMARK = "ml-audit-bench"
SEED = _env_int("SEED", 42)
MAX_STEPS = _env_int("MAX_STEPS", 18)
MAX_EPISODES = _env_int("MAX_EPISODES", 1)
MAX_TOKENS = _env_int("MAX_TOKENS", 500)
TEMPERATURE = _env_float("TEMPERATURE", 0.0)
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 8)
TASK_FILTER = (os.getenv("TASK_FILTER") or "").strip().lower()
DRY_RUN = (os.getenv("DRY_RUN") or "0").strip() == "1"
RETRY_DELAYS = [0.5, 1.0, 2.0]

SYSTEM_PROMPT = (
    "You are an ML integrity auditor. Return exactly one JSON action object. "
    "Valid actions: inspect, compare, flag, unflag, submit. "
    "Do not return prose or markdown."
)

ACTION_PRIORITY = [
    "dataset_info",
    "preprocessing",
    "split_config",
    "model_config",
    "validation_strategy",
    "eval_report",
    "run_history",
    "experiment_notes",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _one_line(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if text else "null"


def _http_request(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if requests is None:
        if REQUESTS_IMPORT_ERROR is not None:
            print(f"requests import unavailable: {REQUESTS_IMPORT_ERROR}", file=sys.stderr, flush=True)
        return None

    url = f"{ENV_URL.rstrip('/')}{endpoint}"
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            else:
                response = requests.post(url, params=params, json=json_body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"ENV request failed for {endpoint}: {exc}", file=sys.stderr, flush=True)
    return None


def _parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    raw = (response_text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "type" in parsed:
                return parsed
            nested = parsed.get("action")
            if isinstance(nested, dict) and "type" in nested:
                return nested
    except json.JSONDecodeError:
        pass

    if "```" in raw:
        for index, chunk in enumerate(raw.split("```")):
            if index % 2 == 0:
                continue
            block = chunk.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "type" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    for candidate in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _fallback_action(observation: Dict[str, Any], step_index: int, budget: int) -> Dict[str, Any]:
    available = [str(a) for a in (observation.get("available_artifacts") or [])]
    inspected = {str(a) for a in (observation.get("inspected_artifacts") or [])}

    for artifact in ACTION_PRIORITY:
        if artifact in available and artifact not in inspected:
            return {"type": "inspect", "artifact": artifact}

    for artifact in available:
        if artifact not in inspected:
            return {"type": "inspect", "artifact": artifact}

    if step_index >= max(0, budget - 2):
        return {"type": "submit", "verdict": "reject", "summary": "No additional evidence available"}

    if available:
        return {"type": "inspect", "artifact": available[0]}

    return {"type": "submit", "verdict": "pass", "summary": "No artifacts returned by environment"}


def _normalize_action(action: Dict[str, Any], observation: Dict[str, Any], step_index: int, budget: int) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return _fallback_action(observation, step_index, budget)

    action_type = str(action.get("type", "")).strip().lower()
    if not action_type:
        return _fallback_action(observation, step_index, budget)

    if action_type in {"load_artifact", "inspect_artifact", "read_artifact"}:
        action_type = "inspect"

    if action_type == "inspect":
        artifact = action.get("artifact") or action.get("artifact_name") or action.get("name")
        if artifact:
            return {"type": "inspect", "artifact": str(artifact)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "compare":
        artifact_a = action.get("artifact_a")
        artifact_b = action.get("artifact_b")
        if artifact_a and artifact_b and str(artifact_a) != str(artifact_b):
            return {"type": "compare", "artifact_a": str(artifact_a), "artifact_b": str(artifact_b)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "flag":
        violation_type = action.get("violation_type")
        evidence_artifact = action.get("evidence_artifact")
        evidence_quote = action.get("evidence_quote")
        severity = action.get("severity") or "medium"
        if violation_type and evidence_artifact and evidence_quote:
            return {
                "type": "flag",
                "violation_type": str(violation_type),
                "evidence_artifact": str(evidence_artifact),
                "evidence_quote": str(evidence_quote),
                "severity": str(severity),
            }
        return _fallback_action(observation, step_index, budget)

    if action_type == "unflag":
        flag_id = action.get("flag_id")
        if flag_id:
            return {"type": "unflag", "flag_id": str(flag_id)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "submit":
        verdict = action.get("verdict") or "reject"
        summary = action.get("summary") or "Submitting audit result"
        return {"type": "submit", "verdict": str(verdict), "summary": str(summary)}

    return _fallback_action(observation, step_index, budget)


def _build_messages(observation: Dict[str, Any], step_index: int, budget: int) -> List[Dict[str, str]]:
    content = {
        "step": step_index + 1,
        "budget": budget,
        "available_artifacts": observation.get("available_artifacts", []),
        "inspected_artifacts": observation.get("inspected_artifacts", []),
        "flags_raised": observation.get("flags_raised", []),
        "last_action_result": observation.get("last_action_result", ""),
        "last_action_error": observation.get("last_action_error") or "none",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Choose the next action as one JSON object only. Current state:\n"
            + json.dumps(content, ensure_ascii=True),
        },
    ]


def _llm_call(client: Any, messages: List[Dict[str, str]]) -> str:
    if client is None:
        return ""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"LLM request failed: {exc}", file=sys.stderr, flush=True)
    return ""


def run_episode(client: Any, task: str, seed: int = 42) -> float:
    step_rewards: List[float] = []
    final_score = 0.0
    total_steps = 0
    response_text = ""

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        reset_data = _http_request("POST", "/reset", params={"task": task, "seed": seed})
        if reset_data is None:
            return 0.0

        observation = reset_data.get("observation", reset_data)
        if not isinstance(observation, dict):
            return 0.0

        budget = _env_int("MAX_STEPS", MAX_STEPS)
        try:
            budget = min(budget, int(observation.get("step_budget", budget)))
        except (TypeError, ValueError):
            pass
        if budget <= 0:
            budget = MAX_STEPS

        done = False
        connection_failed = False

        for step_index in range(budget):
            response_text = ""
            if DRY_RUN:
                action = _fallback_action(observation, step_index, budget)
            else:
                messages = _build_messages(observation, step_index, budget)
                response_text = _llm_call(client, messages)
                parsed = _parse_action(response_text)
                action = parsed if parsed is not None else _fallback_action(observation, step_index, budget)

            action = _normalize_action(action, observation, step_index, budget)

            result = _http_request("POST", "/step", json_body={"action": action})
            if result is None:
                connection_failed = True
                break

            observation = result.get("observation", result)
            if not isinstance(observation, dict):
                connection_failed = True
                break

            reward = _to_float(result.get("reward", 0.0), default=0.0)
            done = bool(result.get("done", False))
            error_value = observation.get("last_action_error")
            error_text = _one_line(error_value)

            total_steps = step_index + 1
            step_rewards.append(reward)
            action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
            print(
                f"[STEP] step={total_steps} action={action_str} reward={reward:.2f} "
                f"done={str(done).lower()} error={error_text}",
                flush=True,
            )

            if done:
                info = result.get("info", {})
                if isinstance(info, dict):
                    final_score = _to_float(info.get("score", 0.0), default=0.0)
                break

        if not done and not connection_failed:
            submit_action = {
                "type": "submit",
                "verdict": "reject",
                "summary": "Reached step budget",
            }
            submit_result = _http_request("POST", "/step", json_body={"action": submit_action})
            if isinstance(submit_result, dict):
                submit_obs = submit_result.get("observation", submit_result)
                submit_reward = _to_float(submit_result.get("reward", 0.0), default=0.0)
                submit_done = bool(submit_result.get("done", False))
                submit_error = _one_line(
                    submit_obs.get("last_action_error") if isinstance(submit_obs, dict) else None
                )

                total_steps += 1
                step_rewards.append(submit_reward)
                submit_str = json.dumps(submit_action, separators=(",", ":"), ensure_ascii=True)
                print(
                    f"[STEP] step={total_steps} action={submit_str} reward={submit_reward:.2f} "
                    f"done={str(submit_done).lower()} error={submit_error}",
                    flush=True,
                )

                info = submit_result.get("info", {})
                if isinstance(info, dict):
                    final_score = _to_float(info.get("score", 0.0), default=0.0)

        return final_score

    except Exception as exc:
        print(f"Episode error for task={task}: {exc}", file=sys.stderr, flush=True)
        if response_text:
            snippet = response_text[:200].replace("\n", " ")
            print(f"Last model response: {snippet}", file=sys.stderr, flush=True)
        return 0.0

    finally:
        try:
            success = (final_score or 0.0) > 0.0
            rewards_str = ",".join(f"{float(r):.2f}" for r in (step_rewards or [])) or "0.00"
            print(
                f"[END] success={str(success).lower()} "
                f"steps={int(total_steps or 0)} "
                f"score={float(final_score or 0.0):.3f} "
                f"rewards={rewards_str}",
                flush=True,
            )
        except Exception:
            print("[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)


def _resolve_tasks() -> List[str]:
    tasks = ["easy", "medium", "hard"]
    if not TASK_FILTER:
        return tasks
    if TASK_FILTER in tasks:
        return [TASK_FILTER]
    print(
        f"Invalid TASK_FILTER={TASK_FILTER!r}; expected one of easy|medium|hard. Running all tasks.",
        file=sys.stderr,
        flush=True,
    )
    return tasks


def main() -> int:
    print("Starting ML Audit inference...", file=sys.stderr, flush=True)

    client: Any = None
    if OpenAI is None:
        if OPENAI_IMPORT_ERROR is not None:
            print(f"WARN: OpenAI import unavailable: {OPENAI_IMPORT_ERROR}", file=sys.stderr, flush=True)
    elif not DRY_RUN and API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            print(f"Failed to initialize OpenAI client: {exc}", file=sys.stderr, flush=True)
            client = None
    elif not DRY_RUN and not API_KEY:
        print("No API key provided; using fallback policy without LLM calls.", file=sys.stderr, flush=True)

    for task in _resolve_tasks():
        for episode_index in range(MAX_EPISODES):
            try:
                run_episode(client, task=task, seed=SEED + episode_index)
            except Exception as exc:
                print(f"Unhandled run_episode failure: {exc}", file=sys.stderr, flush=True)
                continue

    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"Fatal main error: {exc}", file=sys.stderr, flush=True)
    sys.exit(0)