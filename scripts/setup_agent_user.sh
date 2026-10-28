#!/bin/bash
# One-time setup of the macOS user that agent CLIs run as (models with `isolation: agent_user`;
# src/nabench/agentuser.py). Run from the harness user's account:  sudo scripts/setup_agent_user.sh
# Safe to re-run. The runner checks the result before every agent-user run.
set -euo pipefail

AGENT=nabagent                      # sandbox.agent_user
SHARED=/Users/Shared/nab-agent      # holds sandbox.agent_root and the tools on sandbox.agent_path
PACKAGES=(numpy scipy pandas scikit-learn statsmodels h5py nibabel nilearn torch)  # tools.generic of every spec
HARNESS=${SUDO_USER:?run with sudo from the harness account}
HARNESS_HOME=$(dscl . -read "/Users/$HARNESS" NFSHomeDirectory | awk '{print $2}')
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

# 1. A hidden standard user. Its random password is never kept: it is only ever used through sudo.
if ! id "$AGENT" >/dev/null 2>&1; then
  sysadminctl -addUser "$AGENT" -fullName "nabench agent" -shell /bin/zsh -home "/Users/$AGENT" \
    -password "$(openssl rand -base64 32)"
  createhomedir -c -u "$AGENT" >/dev/null
  dscl . -create "/Users/$AGENT" IsHidden 1
fi

# 2. The harness user may run commands as the agent user without a password (never the reverse).
printf '%s ALL=(%s) NOPASSWD: ALL\n' "$HARNESS" "$AGENT" > /etc/sudoers.d/nabench
chmod 440 /etc/sudoers.d/nabench
visudo -cf /etc/sudoers.d/nabench >/dev/null

# 3. The agent user cannot enter the harness user's home: the repository, data, runs and truth roots, other
#    projects and the harness user's own CLI configurations. (The home's group, staff, includes every standard user.)
if ! ls -led "$HARNESS_HOME" | grep -q "user:$AGENT deny"; then
  chmod +a "user:$AGENT deny list,search,readattr,readextattr,readsecurity" "$HARNESS_HOME"
fi

# 4. Read-only tools for the agent: a copy of the harness user's current Claude CLI (a copy cannot update
#    itself, so the recorded version stays the one that runs) and a Python environment with the generic tools.
install -d -o "$HARNESS" -g staff -m 755 "$SHARED" "$SHARED/bin" "$SHARED/run"   # run/: each run's home (sandbox.agent_home)
install -d -o "$HARNESS" -g staff -m 711 "$SHARED/workspaces"
install -o "$HARNESS" -g staff -m 755 "$(readlink -f "$HARNESS_HOME/.local/bin/claude")" "$SHARED/bin/claude"
sudo -u "$HARNESS" -H /opt/homebrew/bin/uv venv --quiet --allow-existing --python /opt/homebrew/bin/python3.12 "$SHARED/venv"
sudo -u "$HARNESS" -H /opt/homebrew/bin/uv pip install --quiet --python "$SHARED/venv/bin/python" "${PACKAGES[@]}"

# 5. A keychain for CLIs that keep their sign-in there (Claude). A user that never logs in to the desktop has none,
#    and macOS will not unlock one named "login" with an empty password; the runner makes it the default for each run.
KC="/Users/$AGENT/Library/Keychains/nabench.keychain-db"
if ! sudo -u "$AGENT" /bin/test -e "$KC"; then
  sudo -u "$AGENT" -H /usr/bin/security create-keychain -p "" "$KC" </dev/null
  sudo -u "$AGENT" -H /usr/bin/security set-keychain-settings "$KC" </dev/null   # never locks on its own
fi

cat <<EOF
Done. Now sign each CLI in as the agent user, from the harness account (not with sudo in front):
  python scripts/agent_login.py --model opus-5.5-cli
  python scripts/agent_login.py --model sol-6.1-cli
EOF
