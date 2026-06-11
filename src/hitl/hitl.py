"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 12: Confidence Router
  TODO 13: Design 3 HITL decision points
"""
from dataclasses import dataclass



HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # Normalize action_type to make matching more robust
        normalized_action = action_type.lower().strip()

        # 1. High-risk actions always require immediate human escalation
        if normalized_action in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        # 2. High confidence: auto-send
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        # 3. Medium confidence: queue for review
        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        # 4. Low confidence: escalate immediately
        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


hitl_decision_points = [
    {
        "id": 1,
        "name": "High-value money transfer approval",
        "trigger": (
            "The user requests a money transfer, especially a large transfer, "
            "international transfer, or transfer to a new beneficiary."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "User identity verification status, transfer amount, recipient details, "
            "account balance, fraud-risk score, recent transaction history, and the "
            "agent's proposed response/action."
        ),
        "example": (
            "A customer asks: 'Transfer 50,000 USD to this new overseas account today.' "
            "Even if the agent is confident, a human reviewer must approve or reject "
            "the action before execution."
        ),
    },
    {
        "id": 2,
        "name": "Account closure or personal information change",
        "trigger": (
            "The user asks to close an account, change password, update phone number, "
            "change email, update address, or modify sensitive personal information."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Authentication result, account ownership evidence, requested change, "
            "previous contact information, risk flags, and whether the request matches "
            "normal user behavior."
        ),
        "example": (
            "A customer says: 'Change the phone number on my account and reset my password.' "
            "Because this can enable account takeover, the request must be escalated to "
            "a human support officer."
        ),
    },
    {
        "id": 3,
        "name": "Ambiguous complaint, fraud claim, or policy exception",
        "trigger": (
            "The agent has medium or low confidence, or the user reports fraud, disputes "
            "a transaction, requests a fee waiver, or asks for an exception to bank policy."
        ),
        "hitl_model": "human-as-tiebreaker",
        "context_needed": (
            "Conversation history, disputed transaction details, relevant bank policy, "
            "customer tier, prior support tickets, confidence score, and alternative "
            "responses generated by the agent."
        ),
        "example": (
            "A customer says: 'I did not make this card transaction, but I also lost my phone "
            "yesterday.' The agent should not make a final judgment alone; a human reviewer "
            "should decide the next step."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
