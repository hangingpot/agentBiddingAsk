---
name: code-review
description: Review code changes for correctness, security risks, maintainability, and test coverage. Use when the user explicitly asks for code-review (for example using the term "code-review") and expects a structured checklist with severity levels.
---

# Code Review

## When To Apply

Apply this skill only when the user explicitly requests `code-review`.

## Output Contract

- Use severity levels: `Blocker`, `High`, `Medium`, `Low`.
- List findings first, ordered by severity.
- Keep each finding concise and actionable.
- Include file paths and symbols when possible.
- If no findings, state "No critical findings" and list residual risks/testing gaps.

## Review Checklist

### 1) Correctness

- [ ] Business logic matches expected behavior
- [ ] Edge cases are handled (null/empty/boundary inputs)
- [ ] Error paths return safe and consistent results
- [ ] State transitions and async flows avoid race conditions
- [ ] Data mapping/serialization is lossless and type-safe

### 2) Security

- [ ] Input validation and output encoding are sufficient
- [ ] No injection vectors (SQL/command/template)
- [ ] Authorization checks exist for protected operations
- [ ] Sensitive data is not leaked in logs/responses
- [ ] Secrets/tokens/credentials are not hardcoded

### 3) Maintainability

- [ ] Naming is clear and consistent with project conventions
- [ ] Functions/classes have focused responsibility
- [ ] Complexity is reasonable; avoid deeply nested logic
- [ ] Duplication is minimized; shared logic is extracted
- [ ] Comments explain "why" for non-obvious decisions

### 4) Tests

- [ ] New/changed behavior has test coverage
- [ ] Regression tests exist for fixed defects
- [ ] Tests assert critical branches and failure cases
- [ ] Mocks/stubs are realistic and minimal
- [ ] Test names describe behavior clearly

## Response Template

Use this structure:

```markdown
## Findings

- **[Blocker]** <short title>
  - Location: `<path>` (`<symbol>`)
  - Risk: <what can break>
  - Fix: <specific change>

- **[High]** <short title>
  - Location: `<path>` (`<symbol>`)
  - Risk: <impact>
  - Fix: <specific change>

## Residual Risks / Test Gaps

- <risk or "None">
```

## Additional Guidance

- Prefer concrete evidence over style-only feedback.
- Do not suggest broad refactors unless they are required to fix risk.
- Keep recommendations directly implementable.
- For deeper standards and examples, read [reference.md](reference.md).
