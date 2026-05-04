# Git Workflow

This file defines the default repository update policy for this project.

## Default Rule

- after any completed user-requested task that changes tracked files, run the
  relevant validation, then commit and push to `origin/main` by default
- do not wait for an extra reminder to push unless the user explicitly says
  not to push yet
- apply the same rule to code changes, documentation changes, and repo
  configuration changes
- do not create empty commits when no tracked files changed

## Pre-Push Gates

- review `git status` and `git diff` before committing
- run the most relevant validation for the touched code paths
- do not commit broken or half-finished work unless the user explicitly asks
  for a checkpoint commit
- confirm that no secrets, credentials, local `.env` files, OCI session
  material, raw EHR data, generated outputs, or local environments are staged

## Commit Rules

- use a short descriptive commit message
- keep commits scoped to the task that was just completed
- if an unrelated local change exists, do not revert it; work around it or ask
  the user if it blocks the task

## Push Rules

- push to `origin/main` by default unless the user requests a different branch
- if `git push` succeeds but the local tracking ref fails to update, verify the
  remote head with `git ls-remote origin refs/heads/main`
- report the pushed commit hash in the final update

## Never

- never force-push, rewrite history, or delete remote branches unless the user
  explicitly requests it
- never push raw data, derived patient data, secrets, or local credential files
- never skip validation on substantive code changes when a relevant check is
  available

## If Blocked

- if push or validation is blocked by authentication, network, or permission
  issues, make the minimum required recovery step and report the blocker
- if the task is complete but push cannot be finished, clearly state whether
  the commit exists locally, whether the remote changed, and what remains
