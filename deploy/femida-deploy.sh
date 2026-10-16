#!/bin/bash
# Forced command behind the GitHub Actions deploy keys on German. Installed as
# /usr/local/sbin/femida-deploy; authorized_keys pins each key to one target:
#
#   restrict,command="/usr/local/sbin/femida-deploy dev"  ssh-ed25519 ... gha-femida-deploy-dev
#   restrict,command="/usr/local/sbin/femida-deploy prod" ssh-ed25519 ... gha-femida-deploy-prod
#
# The client's ssh command line arrives in SSH_ORIGINAL_COMMAND and must be a
# full commit sha; nothing else from the client is honoured. The checkout is
# moved to that sha, the frontend image is rebuilt from it and the stack is
# brought up. If /login does not answer within a minute the previous image and
# checkout are restored.
set -euo pipefail

target=${1:?usage: femida-deploy dev|prod [sha]}
case "$target" in
dev)
  dir=/femida-dev
  compose=(docker-compose -f docker-compose.dev.yml)
  build=(frontend seed)
  health=http://10.0.0.2:8092/login
  ;;
prod)
  dir=/femida
  compose=(docker-compose)
  build=(frontend)
  health=http://10.0.0.2:8082/login
  ;;
*)
  echo "unknown target '$target', expected dev or prod" >&2
  exit 2
  ;;
esac

sha=${SSH_ORIGINAL_COMMAND:-${2:-}}
if ! [[ $sha =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected a full 40-hex commit sha as the ssh command, got '$sha'" >&2
  exit 2
fi

exec > >(tee -a /var/log/femida-deploy.log) 2>&1
echo "$(date -Is) $target -> $sha (client ${SSH_CONNECTION:-local})"

cd "$dir"
prev_sha=$(git rev-parse --verify -q HEAD || true)
prev_image=""
image_name=""
cid=$("${compose[@]}" ps -q frontend 2>/dev/null || true)
if [[ -n $cid ]]; then
  prev_image=$(docker inspect "$cid" --format '{{.Image}}')
  image_name=$(docker inspect "$cid" --format '{{.Config.Image}}')
fi

healthy() {
  local i
  for ((i = 1; i <= 60; i++)); do
    if curl -fsS -o /dev/null --max-time 3 "$health"; then
      echo "$health answered after ${i}s"
      return 0
    fi
    sleep 1
  done
  return 1
}

rollback() {
  echo "!!! $health did not answer, rolling back"
  "${compose[@]}" logs --tail 40 frontend || true
  if [[ -n $prev_image ]]; then
    docker tag "$prev_image" "$image_name"
  fi
  if [[ -n $prev_sha ]]; then
    git -c advice.detachedHead=false checkout --detach "$prev_sha"
  fi
  "${compose[@]}" up -d --no-build
  echo "$(date -Is) $target rolled back to ${prev_sha:-nothing}"
  exit 1
}

depth=()
if [[ -f .git/shallow ]]; then
  depth=(--depth 1)
fi
git fetch "${depth[@]}" origin "$sha"
# Hand edits on the server make this refuse instead of being overwritten.
git -c advice.detachedHead=false checkout --detach "$sha"
"${compose[@]}" build "${build[@]}"
"${compose[@]}" up -d --no-build
healthy || rollback
echo "$(date -Is) $target deployed $sha"
