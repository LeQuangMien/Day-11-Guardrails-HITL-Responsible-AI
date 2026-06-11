"""
Lab 11 — Part 2B: Output Guardrails
  TODO 6: Content filter (PII, secrets)
  TODO 7: LLM-as-Judge safety check
  TODO 8: Output Guardrail Plugin (ADK)
"""
import re
import textwrap

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    # PII / secret patterns to check
    PII_PATTERNS = {
        # Vietnamese phone number, e.g. 0901234567, 84901234567, +84901234567
        "vn_phone_number": r"\b(?:\+?84|0)\d{9,10}\b",

        # Email address
        "email": r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b",

        # Vietnamese National ID / Citizen ID: 9 digits or 12 digits
        "national_id": r"\b(?:\d{9}|\d{12})\b",

        # API key-like secret, e.g. sk-xxxx
        "api_key": r"\bsk-[a-zA-Z0-9_-]+\b",

        # Password assignment, e.g.
        # password: admin123
        # password=admin123
        # password is admin123
        "password": r"\bpassword\s*(?:is|:|=)\s*\S+",

        # Token-like secret, e.g.
        # token: abc123
        # access_token=abc123
        # secret_token is abc123
        "token": r"\b(?:access_token|secret_token|api_token|token)\s*(?:is|:|=)\s*\S+",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""
safety_judge_agent = llm_agent.LlmAgent(
    model="gemini-2.0-flash",
    name="safety_judge",
    instruction=SAFETY_JUDGE_INSTRUCTION,
)

judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


_init_judge()


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Run deterministic content filter first
        filter_result = content_filter(response_text)

        if not filter_result["safe"]:
            self.redacted_count += 1
            response_text = filter_result["redacted"]
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            )

        # 2. Optionally run LLM-as-Judge safety check
        if self.use_llm_judge:
            judge_result = await llm_safety_check(response_text)

            if not judge_result["safe"]:
                self.blocked_count += 1

                safe_message = (
                    "I can't provide that response because it may contain unsafe, "
                    "sensitive, or unreliable information. Please ask a banking-related "
                    "question that does not involve private data, secrets, or harmful instructions."
                )

                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=safe_message)],
                )

        # 3. Return possibly modified response
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses."""
    test_responses = [
        "The 12-month savings rate is 5.5% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


if __name__ == "__main__":


    test_content_filter()
