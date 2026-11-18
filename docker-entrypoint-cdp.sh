#!/bin/sh
# CDP attach entrypoint. No Xvfb and no Chromium: agent-chrome is already running.
set -eu

if [ -z "${LINKEDIN_HTTP_TOKEN:-}" ]; then
  echo "LINKEDIN_HTTP_TOKEN is required for HTTP transport" >&2
  exit 1
fi

if [ -z "${LINKEDIN_CDP_URL:-}" ]; then
  echo "LINKEDIN_CDP_URL is required (attach to agent-chrome; do not launch Chrome)" >&2
  exit 1
fi

exec python -m linkedin_mcp_server "$@"
