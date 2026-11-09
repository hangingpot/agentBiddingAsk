# Code Review Reference

## Severity Definition

- `Blocker`: Must be fixed before merge. Can cause security incident, data corruption, or major outage.
- `High`: Significant risk; likely to cause user-facing failure or serious regression.
- `Medium`: Moderate risk; may impact maintainability, reliability, or edge-case behavior.
- `Low`: Minor issue; quality improvement with limited immediate risk.

## Finding Writing Standard

Each finding should include:

1. Clear issue statement
2. Exact location (`path` + symbol/function/class)
3. Practical impact/risk
4. Specific and minimal fix suggestion

Avoid:

- Vague comments without evidence
- Pure style nitpicks without risk
- Non-actionable advice

## High-Value Checks By Area

### Correctness

- Off-by-one and boundary handling
- Null/undefined/empty input behavior
- Timezone/date arithmetic consistency
- Transactional integrity across multiple writes
- Idempotency for retryable operations

### Security

- Permission checks before side effects
- Unsafe string interpolation in queries/commands
- Excessive data exposure in APIs
- Missing rate limiting on abuse-prone endpoints
- Insecure defaults and debug flags in production paths

### Maintainability

- Hidden coupling across modules
- Overly broad interfaces and leaky abstractions
- Unclear naming for domain concepts
- Duplicated validation/business rules
- Missing rationale around non-obvious trade-offs

### Tests

- Happy path + failure path coverage
- Assertions on side effects (DB writes/events/logging)
- Contract tests for external integrations
- Stable tests (avoid flakiness and timing races)
- Negative tests for authorization and validation

## Quick No-Finding Template

```markdown
## Findings

No critical findings.

## Residual Risks / Test Gaps

- <if any, list here>
```
