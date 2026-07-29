"""
Calibration Suggestions Engine.
Analyzes closed trades to suggest small, conservative weight adjustments.
NEVER applies changes automatically -- only via explicit approval endpoint.
"""

from dataclasses import dataclass

MIN_SAMPLE = 10
MIN_DIFF_PCT = 10
MAX_ADJUSTMENT = 4


@dataclass
class SuggestionResult:
    suggestions: list[dict]
    rules_with_insufficient_data: list[dict]


def generate_suggestions(closed_trades: list) -> SuggestionResult:
    if len(closed_trades) < MIN_SAMPLE:
        return SuggestionResult(suggestions=[], rules_with_insufficient_data=[{
            "message": f"Only {len(closed_trades)} closed trades available -- need at least {MIN_SAMPLE}",
            "sample_size": len(closed_trades), "needed": MIN_SAMPLE}])

    rule_stats: dict[str, dict] = {}
    for trade in closed_trades:
        snapshot = trade.gates_snapshot or {}
        full_details = snapshot.get("full_gate_details", {})
        for gate_name, gate_data in full_details.items():
            rules = gate_data.get("rules", [])
            for rule in rules:
                key = f"{gate_name}::{rule['name']}"
                if key not in rule_stats:
                    rule_stats[key] = {"gate": gate_name, "rule": rule["name"], "weight": rule.get("weight", 0),
                        "passed_wins": 0, "passed_losses": 0, "failed_wins": 0, "failed_losses": 0, "total": 0}
                stats = rule_stats[key]
                stats["total"] += 1
                passed = rule.get("passed", False)
                outcome = trade.outcome
                if passed and outcome == "win": stats["passed_wins"] += 1
                elif passed and outcome == "loss": stats["passed_losses"] += 1
                elif not passed and outcome == "win": stats["failed_wins"] += 1
                elif not passed and outcome == "loss": stats["failed_losses"] += 1

    suggestions = []
    insufficient = []
    for key, stats in rule_stats.items():
        total_passed = stats["passed_wins"] + stats["passed_losses"]
        total_failed = stats["failed_wins"] + stats["failed_losses"]
        if stats["total"] < MIN_SAMPLE:
            insufficient.append({"gate": stats["gate"], "rule": stats["rule"], "sample_size": stats["total"], "needed": MIN_SAMPLE})
            continue
        wr_passed = (stats["passed_wins"] / total_passed * 100) if total_passed > 0 else 0
        wr_failed = (stats["failed_wins"] / total_failed * 100) if total_failed > 0 else 0
        diff = wr_passed - wr_failed
        current_weight = stats["weight"]
        if diff >= MIN_DIFF_PCT:
            adjustment = min(MAX_ADJUSTMENT, round(diff / 5))
            if adjustment > 0:
                suggestions.append({"gate": stats["gate"], "rule": stats["rule"], "current_weight": current_weight,
                    "suggested_weight": round(current_weight + adjustment, 2),
                    "reasoning": f"{stats['rule']} passed in {wr_passed:.0f}% of winning trades vs {wr_failed:.0f}% of losing trades (sample: {stats['total']}) -- strong predictive signal (+{diff:.0f}pp)",
                    "sample_size": stats["total"]})
        elif diff <= -MIN_DIFF_PCT:
            adjustment = min(MAX_ADJUSTMENT, round(abs(diff) / 5))
            if adjustment > 0:
                new_weight = max(1, current_weight - adjustment)
                suggestions.append({"gate": stats["gate"], "rule": stats["rule"], "current_weight": current_weight,
                    "suggested_weight": round(new_weight, 2),
                    "reasoning": f"{stats['rule']} failed in {wr_failed:.0f}% of winning trades vs {wr_passed:.0f}% of losing trades (sample: {stats['total']}) -- negative predictive signal ({diff:.0f}pp)",
                    "sample_size": stats["total"]})
    return SuggestionResult(suggestions=suggestions, rules_with_insufficient_data=insufficient)
