#!/usr/bin/env bash
#
# First-time publication helper: initialize Git, commit, create public GitHub repo with gh, push main.
# Prerequisites: git, GitHub CLI (https://cli.github.com/), authenticated via `gh auth login`.
#
# Intended to run from a clean or local-only clone. If `origin` already exists or the GitHub repo
# name is taken, adjust remotes or pick another name before re-running.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Working directory: ${ROOT}"

echo "==> Step 1/7: Verify GitHub CLI authentication."
echo "    Command: gh auth status"
if ! gh auth status; then
  echo "ERROR: GitHub CLI is not logged in." >&2
  echo "       Run: gh auth login" >&2
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: Git remote 'origin' already exists. Remove or rename it before creating a new repo." >&2
  echo "       Example: git remote remove origin" >&2
  exit 1
fi

echo "==> Step 2/7: Initialize local Git repository."
echo "    Command: git init"
git init

echo "==> Step 3/7: Stage all files."
echo "    Command: git add ."
git add .

echo "==> Step 4/7: Create initial commit."
echo "    Command: git commit -m \"Initial commit: Long Rollout Stabilization for Multi-physics CFD\""
git commit -m "Initial commit: Long Rollout Stabilization for Multi-physics CFD"

echo "==> Step 5/7: Set default branch name to main."
echo "    Command: git branch -M main"
git branch -M main

echo "==> Step 6/7: Create public GitHub repository and attach remote origin."
echo "    Command: gh repo create physics-gnn-surrogate-long-rollout --public --source=. --remote=origin"
gh repo create physics-gnn-surrogate-long-rollout --public --source=. --remote=origin

echo "==> Step 7/7: Push main to origin."
echo "    Command: git push -u origin main"
git push -u origin main

echo "==> Done: repository physics-gnn-surrogate-long-rollout created (public) and main pushed."
