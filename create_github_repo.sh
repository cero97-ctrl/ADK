#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  ./create_github_repo.sh [REPO_NAME] [visibility] [remote_name] [local_dir]
# Examples:
#  ./create_github_repo.sh                       # uses defaults
#  ./create_github_repo.sh MyRepo private origin /path/to/dir
 
# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
 
REPO_NAME="${1:-ADK}"
VISIBILITY="${2:-public}"   # "public" or "private"
REMOTE_NAME="${3:-origin}"
LOCAL_DIR="${4:-$SCRIPT_DIR}"
 
echo "Repository name: $REPO_NAME"
echo "Visibility: $VISIBILITY"
echo "Remote name: $REMOTE_NAME"
echo "Local dir: $LOCAL_DIR"

# Check gh
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install it first (e.g. sudo apt update && sudo apt install gh -y)."
  exit 1
fi

# Check auth
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  echo "gh is not authenticated. Run 'gh auth login' and authenticate interactively, then re-run this script."
  exit 2
fi

cd "$LOCAL_DIR"

# Ensure git repo
if [ ! -d ".git" ]; then
  echo "Initializing a new git repository in $LOCAL_DIR"
  git init
fi

# Ensure there is a current branch
CURRENT_BRANCH=$(git symbolic-ref --quiet --short HEAD || echo "")
if [ -z "$CURRENT_BRANCH" ]; then
  # prefer main
  if git rev-parse --verify main >/dev/null 2>&1; then
    git checkout main
  else
    git checkout -b main
  fi
fi

# Stage and commit if needed
git add -A
if ! git diff --staged --quiet; then
  git commit -m "Initial commit for ADK project" || true
else
  echo "No changes to commit."
fi

# If repo already exists on GitHub, attach remote. Otherwise create it.
if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  echo "Repository $REPO_NAME already exists on GitHub. Linking as remote '$REMOTE_NAME'."
  # try to get HTTPS clone URL
  CLONE_URL=$(gh repo view "$REPO_NAME" --json cloneUrl --jq '.cloneUrl')
  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    echo "Remote $REMOTE_NAME already exists. Ensuring HTTPS URL to avoid SSH errors..."
    git remote set-url "$REMOTE_NAME" "$CLONE_URL"
  else
    git remote add "$REMOTE_NAME" "$CLONE_URL"
  fi
  echo "Pushing branches to $REMOTE_NAME..."
  git push -u "$REMOTE_NAME" --all
  git push --tags "$REMOTE_NAME" || true
else
  echo "Creating repository $REPO_NAME on GitHub (visibility: $VISIBILITY) and pushing..."
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote="$REMOTE_NAME"
  # Ensure HTTPS remote to avoid SSH errors
  CLONE_URL=$(gh repo view "$REPO_NAME" --json cloneUrl --jq '.cloneUrl')
  git remote set-url "$REMOTE_NAME" "$CLONE_URL"
  git push -u "$REMOTE_NAME" --all
fi

# Summary
echo
echo "Remotes:"
git remote -v

echo
echo "Current branch: $(git branch --show-current)"

echo
# Print repository URL
USER_LOGIN=$(gh api user --jq .login)
if [ -n "$USER_LOGIN" ]; then
  echo "Repository URL: https://github.com/$USER_LOGIN/$REPO_NAME"
fi

echo "Done."
