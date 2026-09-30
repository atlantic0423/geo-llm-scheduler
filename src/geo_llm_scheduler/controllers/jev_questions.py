"""E16 working question templates; not a launched or frozen formal campaign."""

import hashlib
import json

from geo_llm_scheduler.controllers.jev import JevError, Question

QUESTION_VERSION = "e16-v1-working"
OPERATORS = {
    "A1": "Phase-Reassign: change one phase's instance within its current region.",
    "A2": "PD-Path-Reassign: jointly change a job's prefill/decode instance path in its region.",
    "A3": "Job-Region-Relocate: jointly relocate a job's prefill and decode to another region.",
    "A4": "Flow-Congestion-Insert: advance a positive-flow-wait operation in the OS sequence.",
    "A5": "PD-Coupled-Insert: jointly advance a job's prefill/decode with positive flow wait.",
    "A6": "Random-Job-LNS: destroy and reinsert a small random set of jobs.",
    "A7": "Active-Pack: move one operation within legal timing to compress instance active time.",
    "A8": "Peak-Coalition: bounded singleton/coalition timing repair for demand peak reduction.",
}
BUDGETS = {"Low": 3, "Medium": 6, "High": 10}


def questions(*, budget: bool = False, continuation: bool = False) -> dict[str, Question]:
    """Build independent narrow questions; operator/budget share one current state."""
    if continuation:
        if budget:
            raise JevError("Continuation observes the post-step state in a separate request")
        return {
            "continue": Question(
                "noul",
                "Does another MacroSearch step have clear expected optimization value "
                "given the current local-search state and previous feedback?",
                {
                    "true": "Recent improvement or remaining exploitable structural problems "
                    "and enough valid candidate opportunities justify another search step.",
                    "false": "Marginal improvement is negligible, relevant problems are relieved, "
                    "or valid candidates are scarce and further search mainly adds cost.",
                },
            )
        }
    result = {
        "operator": Question(
            "choice",
            "Given the current search state, which one A1-A8 MacroSearch operator "
            "is most likely to improve the current subproblem?",
            OPERATORS,
        )
    }
    if budget:
        result["budget"] = Question(
            "choice",
            "What candidate exact-evaluation budget is justified for the next MacroSearch "
            "invocation, balancing available improvement opportunities and computation cost? "
            "This is a per-invocation candidate budget, not trajectory depth.",
            {
                "Low": "Evaluate at most 3 complete candidates.",
                "Medium": "Evaluate at most 6 complete candidates.",
                "High": "Evaluate at most 10 complete candidates.",
            },
        )
    return result


def question_hash() -> str:
    """Hash wording and code-owned budget mapping for a future formal manifest."""
    value = {
        "version": QUESTION_VERSION,
        "routing": {key: q.payload() for key, q in questions(budget=True).items()},
        "continuation": {key: q.payload() for key, q in questions(continuation=True).items()},
        "budget_mapping": BUDGETS,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
