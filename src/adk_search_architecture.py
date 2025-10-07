# src/adk_search_architecture.py
import os, ast, json, re
from typing import Optional, List
from google.genai import types
from google.adk import agents
from google.adk.agents import callback_context as cbx
from google.adk.models import llm_response as llm_resp
from google.adk.models import llm_request as llm_req
from google.adk.tools.google_search_tool import google_search

ARCHITECTURE_SEARCH_INSTR = """# Competition Task
{task_summary}

# Your Task
Find 1 effective neural network architecture pattern suitable for the above task.

# Requirements
- Search for proven architecture patterns (ResNet, Transformer, MLP, CNN, RNN, Attention, etc.)
- Focus on architectures that work well for the specific data type and task
- Provide architectural structure description (layers, connections, key components)
- Include link to source (paper, blog, GitHub, Kaggle)

Use this JSON schema:
Architecture = {{'architecture_name': str, 'architecture_structure': str, 'source_link': str}}
Return: list[Architecture]

Output rules:
- Output ONLY the JSON array (no explanations, no markdown)
- Do NOT wrap in code fences
- 'architecture_structure' should describe: layer types, connections, key design choices
- Include specific details like "3-layer MLP with dropout" or "ResNet-style with skip connections"
- Source link should be a valid URL (arXiv, GitHub, blog, Kaggle)

Example: [{{"architecture_name":"ResNet-18 for tabular","architecture_structure":"18-layer deep network with residual connections, batch normalization, and skip connections. Input->Conv->BN->ReLU->ResBlock(x8)->AvgPool->FC","source_link":"https://arxiv.org/abs/1512.03385"}}]
"""

def get_architecture_retriever_agent_instruction(context: cbx.ReadonlyContext) -> str:
    """Formats the architecture search prompt with state-backed variables."""
    task_summary = context.state.get("task_summary", "")
    num_arch_candidates = context.state.get("num_arch_candidates", 1)
    base = ARCHITECTURE_SEARCH_INSTR.format(
        task_summary=task_summary,
        num_arch_candidates=num_arch_candidates,
    )
    hint = context.state.get("regen_hint_msg")
    return base + (("\n\n" + hint) if hint else "")

def _get_text_from_response(resp: llm_resp.LlmResponse) -> str:
    """Concatenate text parts from the LLM response."""
    txt = ""
    if resp.content and resp.content.parts:
        for p in resp.content.parts:
            if hasattr(p, "text"):
                txt += p.text or ""
    return txt

def _strip_fences(s: str) -> str:
    """Remove common markdown code fences like ```json, ```python, ``` and even multiple backticks."""
    s = re.sub(r"```+\w*\n", "", s)
    s = re.sub(r"```+", "", s)
    return s

def _extract_json_arrays(s: str):
    """Find candidate substrings that look like JSON arrays of objects: [ {...}, {...}, ... ]"""
    pattern = r"\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\}\s*)*\s*\]"
    return re.findall(pattern, s, flags=re.S)

def get_architecture_candidates(
    callback_context: cbx.CallbackContext,
    llm_response: llm_resp.LlmResponse,
) -> Optional[llm_resp.LlmResponse]:
    """Parse JSON list of architecture candidates."""
    # Rebuild text from response parts
    text = ""
    if llm_response.content and llm_response.content.parts:
        for p in llm_response.content.parts:
            if getattr(p, "text", None):
                text += p.text
    if not text:
        return None

    raw = _strip_fences(text)
    candidates = _extract_json_arrays(raw)

    items = None
    # try each candidate
    for cand in candidates:
        parsed = None
        try:
            parsed = json.loads(cand)
        except Exception:
            try:
                parsed = ast.literal_eval(cand)
            except Exception:
                parsed = None
        if isinstance(parsed, list) and parsed:
            items = parsed
            break

    # fallback: smallest bracket region
    if items is None:
        opens = [m.start() for m in re.finditer(r"\[", raw)]
        closes = [m.start() for m in re.finditer(r"\]", raw)]
        for i in opens:
            for j in closes:
                if j > i and (j - i) <= 20000:
                    cand = raw[i:j+1]
                    parsed = None
                    try:
                        parsed = json.loads(cand)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(cand)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, list) and parsed:
                        items = parsed
                        break
            if items is not None:
                break

    if items is None:
        # escalate hints with retry counter
        retry = int(callback_context.state.get("retry_count", 0) or 0) + 1
        callback_context.state["retry_count"] = retry
        if retry == 1:
            hint = "Output ONLY a pure JSON array; no prose, no code fences."
        elif retry == 2:
            hint = "Return a JSON array only — no markdown, no text, no fences."
        else:
            hint = "Return a SINGLE best architecture as a one-element JSON array (no code fences)."
        callback_context.state["regen_hint_msg"] = hint
        return None

    # Accumulate across rounds and deduplicate
    k = int(callback_context.state.get("num_arch_candidates", 1) or 1)
    existing = callback_context.state.get("retrieved_architectures", []) or []
    combined: List = []
    seen_keys = set()

    def key_of(x):
        try:
            return json.dumps(x, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(x)

    for it in existing:
        key = key_of(it)
        if key not in seen_keys:
            combined.append(it)
            seen_keys.add(key)
            if len(combined) >= k:
                break
    if len(combined) < k:
        for it in items:
            key = key_of(it)
            if key not in seen_keys:
                combined.append(it)
                seen_keys.add(key)
                if len(combined) >= k:
                    break

    callback_context.state["retrieved_architectures"] = combined
    
    # If we have enough, mark as finished
    if len(combined) >= k:
        callback_context.state["init_arch_finish"] = True
    else:
        remaining = k - len(combined)
        callback_context.state["regen_hint_msg"] = (
            f"Return ONLY a JSON array with {remaining} NEW architecture candidate(s) not previously suggested; no markdown, no fences."
        )
        callback_context.state["init_arch_finish"] = False
        return None
    
    # clear regen hint
    try:
        del callback_context.state["regen_hint_msg"]
    except Exception:
        callback_context.state["regen_hint_msg"] = None
    return None

def check_architecture_finish(
    callback_context: cbx.CallbackContext,
    llm_request: llm_req.LlmRequest,
) -> Optional[llm_resp.LlmResponse]:
    """Stop the loop when we have at least k candidates accumulated."""
    if callback_context.state.get("init_arch_finish", False):
        return llm_resp.LlmResponse()
    return None

def make_search_architecture_root_agent(task_summary: str, k: int = 1):
    """
    Build a root agent that searches for neural network architectures suitable for the task.
    """
    def bootstrap_state(callback_context: cbx.CallbackContext) -> Optional[types.Content]:
        callback_context.state["task_summary"] = task_summary
        callback_context.state["num_arch_candidates"] = k
        return None

    def emit_result(callback_context: cbx.CallbackContext) -> Optional[types.Content]:
        items = callback_context.state.get("retrieved_architectures", [])
        k = int(callback_context.state.get("num_arch_candidates", 1) or 1)
        if not items:
            raise RuntimeError("Architecture search failed: no valid architecture candidates retrieved.")
        # Take exactly k (or fewer if somehow not enough)
        selected = items[:k]
        return types.Content(parts=[types.Part(text=json.dumps(selected, ensure_ascii=False))])

    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")

    architecture_retriever_agent = agents.Agent(
        model=model_name,
        name="architecture_retriever_agent",
        description="Retrieve effective neural network architectures + structure descriptions.",
        instruction=get_architecture_retriever_agent_instruction,
        tools=[google_search],
        before_model_callback=check_architecture_finish,
        after_model_callback=get_architecture_candidates,
        generate_content_config=types.GenerateContentConfig(temperature=0.7),
        include_contents="none",
    )

    architecture_retriever_loop = agents.LoopAgent(
        name="architecture_retriever_loop",
        description="Retrieve architectures until parse succeeds.",
        sub_agents=[architecture_retriever_agent],
        max_iterations=10,
    )

    root = agents.SequentialAgent(
        name="architecture_search_root",
        description="Root agent for Architecture Search",
        sub_agents=[architecture_retriever_loop],
        before_agent_callback=bootstrap_state,
        after_agent_callback=emit_result,
    )
    return root

