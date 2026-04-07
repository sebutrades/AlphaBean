"""
simulation/agents/base.py — Base agent interface for Ollama + Anthropic.

Provides async wrappers for both local Ollama (Qwen3) and Anthropic API
calls with JSON response extraction, retry logic, and cost tracking.
"""
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class AgentResponse:
    """Structured response from any agent call."""
    success: bool
    data: dict                # parsed JSON response
    raw_text: str = ""
    model: str = ""
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class CostTracker:
    """Tracks API call costs across the simulation."""
    ollama_calls: int = 0
    haiku_calls: int = 0
    sonnet_calls: int = 0
    haiku_input_tokens: int = 0
    haiku_output_tokens: int = 0
    sonnet_input_tokens: int = 0
    sonnet_output_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD."""
        # Haiku: $0.25/M input, $1.25/M output
        haiku = (self.haiku_input_tokens * 0.25 + self.haiku_output_tokens * 1.25) / 1_000_000
        # Sonnet: $3/M input, $15/M output
        sonnet = (self.sonnet_input_tokens * 3.0 + self.sonnet_output_tokens * 15.0) / 1_000_000
        return round(haiku + sonnet, 4)

    def summary(self) -> str:
        return (
            f"Ollama: {self.ollama_calls} calls | "
            f"Haiku: {self.haiku_calls} calls ({self.haiku_input_tokens + self.haiku_output_tokens:,} tokens) | "
            f"Sonnet: {self.sonnet_calls} calls ({self.sonnet_input_tokens + self.sonnet_output_tokens:,} tokens) | "
            f"Est. cost: ${self.estimated_cost:.2f}"
        )


# Global cost tracker for the simulation
cost_tracker = CostTracker()


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response, handling thinking blocks and fences."""
    # Strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try fenced JSON first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Depth-tracked brace matching
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


async def call_ollama(
    prompt: str,
    model: str = "qwen3:8b",
    system: str = "",
    temperature: float = 0.15,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> AgentResponse:
    """Call local Ollama model asynchronously."""
    t0 = time.time()

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(OLLAMA_URL, json=payload)

        if resp.status_code != 200:
            return AgentResponse(
                success=False, data={}, model=model,
                error=f"Ollama {resp.status_code}: {resp.text[:200]}",
                elapsed=time.time() - t0,
            )

        raw_text = resp.json().get("response", "")
        data = _extract_json(raw_text)

        cost_tracker.ollama_calls += 1

        return AgentResponse(
            success=data is not None,
            data=data or {},
            raw_text=raw_text[:2000],
            model=model,
            elapsed=time.time() - t0,
        )

    except Exception as e:
        return AgentResponse(
            success=False, data={}, model=model,
            error=str(e), elapsed=time.time() - t0,
        )


async def call_anthropic(
    prompt: str,
    system: str = "",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 2048,
    temperature: float = 0.3,
    timeout: int = 60,
) -> AgentResponse:
    """Call Anthropic API asynchronously."""
    if not HAS_ANTHROPIC:
        return AgentResponse(
            success=False, data={}, model=model,
            error="anthropic SDK not installed",
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return AgentResponse(
            success=False, data={}, model=model,
            error="ANTHROPIC_API_KEY not set",
        )

    t0 = time.time()

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text
        data = _extract_json(raw_text)
        elapsed = time.time() - t0

        # Track costs
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        if "haiku" in model:
            cost_tracker.haiku_calls += 1
            cost_tracker.haiku_input_tokens += in_tok
            cost_tracker.haiku_output_tokens += out_tok
        elif "sonnet" in model:
            cost_tracker.sonnet_calls += 1
            cost_tracker.sonnet_input_tokens += in_tok
            cost_tracker.sonnet_output_tokens += out_tok

        return AgentResponse(
            success=data is not None,
            data=data or {},
            raw_text=raw_text[:2000],
            model=model,
            elapsed=elapsed,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    except Exception as e:
        return AgentResponse(
            success=False, data={}, model=model,
            error=str(e), elapsed=time.time() - t0,
        )
