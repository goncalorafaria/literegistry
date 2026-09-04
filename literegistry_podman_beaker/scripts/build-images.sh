#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEFAULT_TAG="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$PACKAGE_ROOT/pyproject.toml" | head -n 1)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: scripts/build-images.sh [IMAGE_REPOSITORY] [IMAGE_TAG]

Builds Redis, gateway, rootless Podman server, Docker mirror, Podman-client
warmup, and TMAX live-fire images from this package directory. IMAGE_REPOSITORY is optional (for example, goncalof or
registry.example/team).

Environment:
  DOCKER_BIN=docker   Docker-compatible command
  PLATFORM=...       Optional target platform
  PULL_BASES=0|1     Pull official base images first (default: 1)
  PIP_FIND_LINKS=...  Optional local or internal Python wheelhouse URL
  PUSH_IMAGES=0|1    Push each successfully built image (default: 0)
USAGE
    exit 0
fi

IMAGE_REPOSITORY="${1:-${IMAGE_REPOSITORY:-}}"
IMAGE_TAG="${2:-${IMAGE_TAG:-$DEFAULT_TAG}}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
PUSH_IMAGES="${PUSH_IMAGES:-0}"
PULL_BASES="${PULL_BASES:-1}"
PLATFORM="${PLATFORM:-}"

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
gateway_image="${repository_prefix}literegistry-podman-gateway:${IMAGE_TAG}"
podman_image="${repository_prefix}literegistry-podman-server:${IMAGE_TAG}"
mirror_image="${repository_prefix}literegistry-docker-mirror:${IMAGE_TAG}"
warmup_image="${repository_prefix}literegistry-podman-warmup:${IMAGE_TAG}"
live_fire_image="${repository_prefix}literegistry-podman-live-fire:${IMAGE_TAG}"

build_image() {
    local dockerfile="$1"
    local image="$2"

    "$DOCKER_BIN" "${build_options[@]}" \
        --file "$PACKAGE_ROOT/docker/$dockerfile" \
        --tag "$image" \
        "$PACKAGE_ROOT"

    if [[ "$PUSH_IMAGES" == "1" ]]; then
        "$DOCKER_BIN" push "$image"
    fi
}

build_image Dockerfile.redis "$redis_image"
build_image Dockerfile.gateway "$gateway_image"
build_image Dockerfile.podman "$podman_image"
build_image Dockerfile.mirror "$mirror_image"
build_image Dockerfile.warmup "$warmup_image"
build_image Dockerfile.live-fire "$live_fire_image"

printf 'REDIS_IMAGE=%s\n' "$redis_image"
printf 'GATEWAY_IMAGE=%s\n' "$gateway_image"
printf 'PODMAN_IMAGE=%s\n' "$podman_image"
printf 'DOCKER_MIRROR_IMAGE=%s\n' "$mirror_image"
printf 'PODMAN_WARMUP_IMAGE=%s\n' "$warmup_image"
printf 'PODMAN_LIVE_FIRE_IMAGE=%s\n' "$live_fire_image"
