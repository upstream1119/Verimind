"""Claim-level, attribution-aware policy primitives for QASPER Gate V2."""

import json
from typing import Dict, List, Sequence


LABELS = {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
ATTRIBUTIONS = {
    "CURRENT_PAPER",
    "RELATED_WORK",
    "BACKGROUND",
    "UNRESOLVED",
}
CLAIM_SUBJECTS = {
    "CURRENT_PAPER",
    "PRIOR_WORK",
    "GENERAL_FIELD",
    "UNRESOLVED",
}
EVIDENCE_OWNERS = {
    "CURRENT_PAPER",
    "PRIOR_WORK",
    "MIXED",
    "UNRESOLVED",
}
ISSUE_TYPES = {
    "unsupported_fact",
    "attribution_error",
    "metric_mismatch",
    "numeric_mismatch",
    "relation_reversal",
    "evidence_missing",
    "evidence_truncated",
    "other",
}
POLICY_VERSION = "gate_v2_majority_reject_single_warning_v1"
PROMPT_VERSION = "gate_v2_claim_attribution_v1_1"


def _strip_json_fence(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def parse_gate_judgment(raw_response: str) -> Dict[str, object]:
    parsed = json.loads(_strip_json_fence(raw_response))
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")

    model_label = str(parsed.get("overall_label", "")).strip().upper()
    if model_label not in LABELS:
        raise ValueError(
            "overall_label must be SUPPORTED, UNSUPPORTED, or UNCERTAIN"
        )
    claims = parsed.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims must be a non-empty list")

    normalized_claims: List[Dict[str, object]] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise ValueError(f"claim {index} must be an object")
        verdict = str(claim.get("verdict", "")).strip().upper()
        if verdict not in LABELS:
            raise ValueError(
                f"claim {index} verdict must be one of {sorted(LABELS)}"
            )
        claim_subject = str(
            claim.get("claim_subject", "")
        ).strip().upper()
        if claim_subject not in CLAIM_SUBJECTS:
            raise ValueError(
                f"claim {index} claim_subject must be one of "
                f"{sorted(CLAIM_SUBJECTS)}"
            )
        evidence_owner = str(
            claim.get("evidence_owner", "")
        ).strip().upper()
        if evidence_owner not in EVIDENCE_OWNERS:
            raise ValueError(
                f"claim {index} evidence_owner must be one of "
                f"{sorted(EVIDENCE_OWNERS)}"
            )
        attribution = str(
            claim.get("attribution", "")
        ).strip().upper()
        if attribution not in ATTRIBUTIONS:
            raise ValueError(
                f"claim {index} attribution must be one of "
                f"{sorted(ATTRIBUTIONS)}"
            )
        issue_types = claim.get("issue_types")
        if not isinstance(issue_types, list):
            raise ValueError(f"claim {index} issue_types must be a list")
        normalized_issues = [
            str(issue).strip().lower()
            for issue in issue_types
            if str(issue).strip()
        ]
        invalid_issues = sorted(set(normalized_issues) - ISSUE_TYPES)
        if invalid_issues:
            raise ValueError(
                f"claim {index} has invalid issue types: {invalid_issues}"
            )
        original_verdict = verdict
        if (
            claim_subject == "CURRENT_PAPER"
            and evidence_owner == "PRIOR_WORK"
        ):
            verdict = "UNSUPPORTED"
            if "attribution_error" not in normalized_issues:
                normalized_issues.append("attribution_error")
        normalized = {
            "claim": str(claim.get("claim", "")).strip(),
            "claim_subject": claim_subject.lower(),
            "verdict": verdict.lower(),
            "verdict_repaired": verdict != original_verdict,
            "evidence": str(claim.get("evidence", "")).strip(),
            "evidence_section": str(
                claim.get("evidence_section", "")
            ).strip(),
            "evidence_owner": evidence_owner.lower(),
            "attribution": attribution.lower(),
            "issue_types": normalized_issues,
            "reason": str(claim.get("reason", "")).strip(),
        }
        required_text = [
            normalized["claim"],
            normalized["evidence"],
            normalized["evidence_section"],
            normalized["reason"],
        ]
        if not all(required_text):
            raise ValueError(
                f"claim {index} requires claim, evidence, "
                "evidence_section, and reason"
            )
        normalized_claims.append(normalized)

    unsupported_count = sum(
        claim["verdict"] == "unsupported"
        for claim in normalized_claims
    )
    uncertain_count = sum(
        claim["verdict"] == "uncertain"
        for claim in normalized_claims
    )
    if unsupported_count:
        canonical_label = "unsupported"
    elif uncertain_count:
        canonical_label = "uncertain"
    else:
        canonical_label = "supported"
    summary_reason = str(parsed.get("summary_reason", "")).strip()
    if not summary_reason:
        raise ValueError("summary_reason must be non-empty")
    return {
        "model_label": model_label.lower(),
        "label": canonical_label,
        "label_repaired": model_label.lower() != canonical_label,
        "claims": normalized_claims,
        "unsupported_claim_count": unsupported_count,
        "uncertain_claim_count": uncertain_count,
        "summary_reason": summary_reason,
    }


def aggregate_judgments(labels: Sequence[str]) -> Dict[str, object]:
    normalized = [str(label).strip().lower() for label in labels]
    invalid = sorted(
        set(normalized) - {"supported", "unsupported", "uncertain"}
    )
    if invalid:
        raise ValueError(f"Invalid judge labels: {invalid}")
    unsupported_votes = normalized.count("unsupported")
    uncertain_votes = normalized.count("uncertain")
    supported_votes = normalized.count("supported")
    if unsupported_votes >= 2:
        decision = "REJECT"
    elif unsupported_votes == 1 or uncertain_votes >= 1:
        decision = "WARNING"
    else:
        decision = "ANSWER"
    return {
        "policy_version": POLICY_VERSION,
        "decision": decision,
        "unsupported_votes": unsupported_votes,
        "uncertain_votes": uncertain_votes,
        "supported_votes": supported_votes,
    }


def make_gate_prompt(question: str, answer: str, context: str) -> str:
    schema = {
        "overall_label": "SUPPORTED, UNSUPPORTED, or UNCERTAIN",
        "claims": [
            {
                "claim": "one material factual claim from the answer",
                "claim_subject": (
                    "CURRENT_PAPER, PRIOR_WORK, GENERAL_FIELD, or UNRESOLVED"
                ),
                "verdict": "SUPPORTED, UNSUPPORTED, or UNCERTAIN",
                "evidence": "short exact evidence or No supporting evidence",
                "evidence_section": "section name or UNRESOLVED",
                "evidence_owner": (
                    "CURRENT_PAPER, PRIOR_WORK, MIXED, or UNRESOLVED"
                ),
                "attribution": (
                    "CURRENT_PAPER, RELATED_WORK, BACKGROUND, or UNRESOLVED"
                ),
                "issue_types": [
                    "zero or more allowed issue type strings"
                ],
                "reason": "brief evidence-grounded explanation",
            }
        ],
        "summary_reason": "brief overall rationale",
    }
    allowed_issues = ", ".join(sorted(ISSUE_TYPES))
    return (
        "Audit the generated answer against the retrieved scientific-paper "
        "context at the atomic-claim level.\n"
        "Use only the retrieved context. Do not use outside knowledge.\n"
        "Decompose the answer into every material factual claim.\n"
        "SUPPORTED means the claim is directly supported or a reasonable "
        "paraphrase.\n"
        "UNSUPPORTED means the claim is absent, contradicted, assigned to "
        "the wrong study or section role, uses the wrong number or metric, "
        "or reverses a relation.\n"
        "UNCERTAIN is reserved for evidence that cannot be resolved because "
        "the context is truncated, internally inconsistent, or contains "
        "placeholders such as INLINEFORM0.\n"
        "Do not mark a merely incomplete answer as unsupported.\n\n"
        "Section-aware attribution rules:\n"
        "1. In the question and answer, 'the paper', 'they', 'the authors', "
        "and an unqualified 'the method/system' refer to CURRENT_PAPER.\n"
        "2. evidence_owner identifies who actually performed the method or "
        "reported the result. Named external authors, cited studies, and "
        "phrases such as prior work or previous studies are PRIOR_WORK, "
        "even when they appear inside a topical section rather than a "
        "section literally titled Related Work.\n"
        "3. Methods or results owned by PRIOR_WORK cannot support a "
        "CURRENT_PAPER claim. Such a conflict must be UNSUPPORTED with "
        "issue_types including attribution_error.\n"
        "4. A baseline mentioned for comparison is not automatically a "
        "method used by the current paper.\n"
        "5. Preserve metric types such as accuracy, F1, recall, WER, "
        "relative improvement, and percentage points.\n"
        "6. Preserve numerical values and comparison direction.\n"
        "7. If a placeholder hides the value needed to verify a claim, use "
        "UNCERTAIN rather than guessing.\n\n"
        f"Allowed issue_types: {allowed_issues}.\n"
        "Return exactly one JSON object and no Markdown or prose outside it."
        "\nSchema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Retrieved Context]\n{context}\n\n"
        f"[Generated Answer]\n{answer}\n"
    )
