"""Missing-receipt budget allowance, never a substitute for provider billing.

Text requests use UTF-8 bytes plus a protocol margin for input, and the sent
output limit (or a deliberately generous fallback) for output. This is a
conservative accounting policy, not a proven bound for an unknown provider.
"""
from tools.JsonInput import loads


POLICY = "text-bytes-output-v1"
PROTOCOL_MARGIN = 4096
UNKNOWN_OUTPUT_ALLOWANCE = 131072


def usage_counts_valid(usage):
    """Partial receipts are allowed, but total cannot contradict a known part."""
    fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if not isinstance(usage, dict) or any(type(usage[k]) is not int or usage[k] < 0
                                          for k in fields if k in usage):
        return False
    if 'total_tokens' not in usage:
        return True
    parts = [usage[k] for k in fields[:2] if k in usage]
    total = usage['total_tokens']
    return total == sum(parts) if len(parts) == 2 else total >= sum(parts)


def missing_usage_reservation(request, usage):
    """Return None for unreadable requests; those still require reconciliation."""
    try:
        body = loads(request, location="token reservation request", expected=dict)
        size = len(request.encode("utf-8"))
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return None
    output = body.get("max_completion_tokens", body.get("max_tokens"))
    if type(output) is not int or output <= 0:
        output = UNKNOWN_OUTPUT_ALLOWANCE
    # Valid partial receipts can raise an allowance, never reduce it.
    partial = {key: value for key, value in usage.items()
               if type(value) is int and value >= 0}
    input_allowance = max(size + PROTOCOL_MARGIN, partial.get("prompt_tokens", 0))
    output_allowance = max(output, partial.get("completion_tokens", 0))
    return max(input_allowance + output_allowance, partial.get("total_tokens", 0))
