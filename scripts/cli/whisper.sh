#!/usr/bin/env bash
#
# whisper.sh — send a YouTube URL or local file to the whisper MCP server.
# Local files are uploaded as raw bytes so they can live anywhere, not just
# under the server's DATA_DIR mount.
#
# Usage:
#   whisper.sh <url-or-path> [--language ru] [--no-cleanup]
#
# Environment:
#   WHISPER_URL   base URL of the server (default: http://localhost:17800)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <url-or-path> [--language CODE] [--no-cleanup]" >&2
    exit 2
fi

input="$1"
shift

language=""
no_cleanup=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --language)
            language="${2:?--language requires a value}"
            shift 2
            ;;
        --language=*)
            language="${1#--language=}"
            shift
            ;;
        --no-cleanup)
            no_cleanup=1
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

server="${WHISPER_URL:-http://localhost:17800}"
server="${server%/}"

urlencode() {
    # POSIX-ish percent-encode using printf + xxd. jq -sRr @uri is shorter
    # but jq may not be installed; fall back to python3 if available.
    if command -v jq >/dev/null 2>&1; then
        printf %s "$1" | jq -sRr @uri
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
    else
        # Last-resort minimal encoder — handles spaces and a few hot chars.
        local s="$1"
        s="${s// /%20}"
        s="${s//&/%26}"
        s="${s//#/%23}"
        s="${s//\?/%3F}"
        printf '%s' "$s"
    fi
}

http_response=$(mktemp)
trap 'rm -f "$http_response"' EXIT

print_result() {
    if command -v jq >/dev/null 2>&1; then
        text=$(jq -r '.text' "$http_response")
        lang=$(jq -r '.language // "?"' "$http_response")
        segments=$(jq -r '.segments' "$http_response")
        elapsed=$(jq -r '.elapsed_ms' "$http_response")
        printf '%s\n' "$text"
        printf '[whisper] lang=%s segments=%s elapsed_ms=%s\n' "$lang" "$segments" "$elapsed" >&2
        maybe_save "$text"
    else
        cat "$http_response"
        echo >&2
        echo '[whisper] install `jq` for prettier output and save-prompt support' >&2
    fi
}

# Prompt the user (only when stdin + stdout are terminals) to save the
# transcript as a .txt file in the current working directory.
maybe_save() {
    local text="$1"
    [[ -t 0 && -t 1 ]] || return 0

    local stem
    if [[ "$input" =~ ^[hH][tT][tT][pP][sS]?:// ]]; then
        stem=$(printf '%s' "$input" | sed -E 's|^.*[/=]||; s|[^A-Za-z0-9._-]|_|g')
        [[ -z "$stem" ]] && stem="transcript-$(date +%Y%m%d-%H%M%S)"
    else
        stem=$(basename -- "$input")
        stem="${stem%.*}"
    fi
    local default="$PWD/$stem.txt"

    printf 'Save transcript to %q? [Y/n/path] ' "$default" >&2
    local answer
    IFS= read -r answer || return 0
    answer="${answer## }"
    answer="${answer%% }"

    local target
    case "$answer" in
        ''|y|Y|yes|YES) target="$default" ;;
        n|N|no|NO)      return 0 ;;
        /*)             target="$answer" ;;
        *)              target="$PWD/$answer" ;;
    esac

    if printf '%s' "$text" > "$target"; then
        printf '[whisper] saved to %s\n' "$target" >&2
    else
        echo "save failed: $target" >&2
        return 1
    fi
}

if [[ "$input" =~ ^[hH][tT][tT][pP][sS]?:// ]]; then
    endpoint="$server/api/transcribe/youtube"
    if [[ $no_cleanup -eq 1 ]]; then cleanup=false; else cleanup=true; fi
    body="{\"url\":\"$input\",\"cleanup\":$cleanup"
    if [[ -n "$language" ]]; then body="$body,\"language\":\"$language\""; fi
    body="$body}"

    http_code=$(curl -sS -o "$http_response" -w '%{http_code}' \
        -X POST "$endpoint" \
        -H 'Content-Type: application/json' \
        --data-binary "$body")
else
    if [[ ! -f "$input" ]]; then
        echo "file not found: $input" >&2
        exit 1
    fi
    filename=$(basename "$input")
    enc_name=$(urlencode "$filename")
    endpoint="$server/api/transcribe/upload?filename=$enc_name"
    if [[ -n "$language" ]]; then
        enc_lang=$(urlencode "$language")
        endpoint="$endpoint&language=$enc_lang"
    fi
    size=$(stat -c '%s' "$input" 2>/dev/null || stat -f '%z' "$input")
    echo "[whisper] uploading $input (${size} bytes)" >&2

    http_code=$(curl -sS -o "$http_response" -w '%{http_code}' \
        -X POST "$endpoint" \
        -H 'Content-Type: application/octet-stream' \
        --data-binary "@$input")
fi

if [[ "$http_code" != "200" ]]; then
    echo "whisper request failed: HTTP $http_code" >&2
    cat "$http_response" >&2
    echo >&2
    exit 1
fi

print_result
