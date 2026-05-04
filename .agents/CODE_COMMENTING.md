# Code Commenting Skill

This file defines the default commenting and docstring style for code in this
repository.

## Goal

- make the code readable to a new human collaborator without forcing them to
  reverse-engineer intent from implementation details
- explain why the code exists, what contract it follows, and where the risky
  edges are

## Required Defaults

- add a short module docstring to each non-trivial Python module
- add docstrings to public functions, public classes, and any helper whose
  purpose is not obvious from its name alone
- prefer comments that explain intent, assumptions, protocol quirks, or
  research constraints
- keep comments short and factual

## Avoid

- do not add comments that merely restate the next line of code
- do not narrate simple assignments, loops, or conditionals
- do not write vague comments such as `# handle stuff` or `# magic`
- do not let comments drift out of sync with the implementation

## When Comments Are Expected

- authentication flows and request signing logic
- OCI or provider-specific compatibility workarounds
- fallback logic across model families or API parameter variants
- validation gates, safety checks, and failure modes
- research-specific assumptions that affect reproducibility

## Docstring Style

- first line: one-sentence summary
- add a second short paragraph only when side effects, constraints, or return
  behavior are not obvious
- keep docstrings compact; optimize for fast scanning

## Inline Comment Style

- comment the reason, not the mechanics
- place comments immediately above the relevant block when possible
- use inline end-of-line comments only when a very short clarification is best

## Maintenance Rule

- when modifying existing code, update stale docstrings and comments in the
  touched area as part of the same change
