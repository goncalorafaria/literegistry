#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEFAULT_TAG="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$PACKAGE_ROOT/pyproject.toml" | head -n 1)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: scripts/build-images.sh [IMAGE_REPOSITORY] [IMAGE_TAG]

Builds the Redis, shared services, terminal, and vLLM images from this package directory.
Optionally rebuilds the canonical JTC Lucene image from a JTC checkout.

Environment:
  DOCKER_BIN=docker        Docker-compatible command
  PLATFORM=...             Optional target platform
  PULL_BASES=0|1           Pull official base images first (default: 1)
  PIP_FIND_LINKS=...       Optional local or internal Python wheelhouse URL
  PUSH_IMAGES=0|1          Push each successfully built image (default: 0)
  BUILD_LOCAL_SEARCH=0|1   Also build the JTC Lucene image (default: 0)
  JTC_BUILD_CONTEXT=...    JTC checkout root; required with BUILD_LOCAL_SEARCH=1
  VLLM_BASE_IMAGE=...      Override vllm/vllm-openai:latest
USAGE
    exit 0
fi

IMAGE_REPOSITORY="${1:-${IMAGE_REPOSITORY:-}}"
IMAGE_TAG="${2:-${IMAGE_TAG:-$DEFAULT_TAG}}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
PUSH_IMAGES="${PUSH_IMAGES:-0}"
PULL_BASES="${PULL_BASES:-1}"
PLATFORM="${PLATFORM:-}"
BUILD_LOCAL_SEARCH="${BUILD_LOCAL_SEARCH:-0}"
JTC_BUILD_CONTEXT="${JTC_BUILD_CONTEXT:-}"

if [[ -z "$IMAGE_TAG" ]]; then
    echo "could not determine image tag" >&2
    exit 2
fi
if ! command -v "$DOCKER_BIN" >/dev/null 2>&1; then
    echo "Docker command not found: $DOCKER_BIN" >&2
    exit 127
fi

repository_prefix=""
if [[ -n "$IMAGE_REPOSITORY" ]]; then
    repository_prefix="${IMAGE_REPOSITORY%/}/"
fi

build_options=(build)
if [[ "$PULL_BASES" == "1" ]]; then
    build_options+=(--pull)
fi
if [[ -n "$PLATFORM" ]]; then
    build_options+=(--platform "$PLATFORM")
fi
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    build_options+=(--build-arg PIP_INDEX_URL)
fi
if [[ -n "${PIP_FIND_LINKS:-}" ]]; then
    build_options+=(--build-arg PIP_FIND_LINKS)
fi
if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
    build_options+=(--build-arg PIP_TRUSTED_HOST)
fi

redis_image="${repository_prefix}literegistry-redis:${IMAGE_TAG}"
services_image="${repository_prefix}literegistry-base-services:${IMAGE_TAG}"
terminal_image="${repository_prefix}literegistry-base-terminal:${IMAGE_TAG}"
vllm_image="${repository_prefix}literegistry-base-vllm:${IMAGE_TAG}"
local_search_image="${repository_prefix}jtc-local-search-lucene-bm25:${IMAGE_TAG}"

build_image() {
    local dockerfile="$1"
    local image="$2"
    shift 2

    "$DOCKER_BIN" "${build_options[@]}" "$@" \
        --file "$PACKAGE_ROOT/docker/$dockerfile" \
        --tag "$image" \
        "$PACKAGE_ROOT"

    if [[ "$PUSH_IMAGES" == "1" ]]; then
        "$DOCKER_BIN" push "$image"
    fi
}

build_image Dockerfile.redis "$redis_image"
build_image Dockerfile.services "$services_image"
build_image Dockerfile.terminal "$terminal_image"
vllm_args=()
if [[ -n "${VLLM_BASE_IMAGE:-}" ]]; then
    vllm_args+=(--build-arg "VLLM_BASE_IMAGE=$VLLM_BASE_IMAGE")
fi
build_image Dockerfile.vllm "$vllm_image" "${vllm_args[@]}"

if [[ "$BUILD_LOCAL_SEARCH" == "1" ]]; then
    if [[ -z "$JTC_BUILD_CONTEXT" ]]; then
        echo "JTC_BUILD_CONTEXT is required when BUILD_LOCAL_SEARCH=1" >&2
        exit 2
    fi
    if [[ ! -d "$JTC_BUILD_CONTEXT/search" ]]; then
        echo "JTC_BUILD_CONTEXT must contain search/: $JTC_BUILD_CONTEXT" >&2
        exit 2
    fi
    "$DOCKER_BIN" "${build_options[@]}" \
        --file "$PACKAGE_ROOT/docker/Dockerfile.local-search" \
        --tag "$local_search_image" \
        "$JTC_BUILD_CONTEXT"
    if [[ "$PUSH_IMAGES" == "1" ]]; then
        "$DOCKER_BIN" push "$local_search_image"
    fi
fi

printf 'REDIS_IMAGE=%s\n' "$redis_image"
printf 'SERVICES_IMAGE=%s\n' "$services_image"
printf 'TERMINAL_IMAGE=%s\n' "$terminal_image"
printf 'VLLM_IMAGE=%s\n' "$vllm_image"
if [[ "$BUILD_LOCAL_SEARCH" == "1" ]]; then
    printf 'LOCAL_SEARCH_IMAGE=%s\n' "$local_search_image"
fi
