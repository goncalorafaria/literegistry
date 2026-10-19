#!/usr/bin/env bash
set -euo pipefail

# Honor the same override when the image is used directly with CLI flags.
# Beaker supervisor commands receive the network choice through the environment.
network="${PODMAN_AFFINITY_NETWORK:-pasta}"
if [[ $# == 0 || "${1:-}" == -* || ( "${1:-}" == literegistry && "${2:-}" == podman ) ]]; then
    previous=""
    for argument in "$@"; do
        if [[ "$previous" == --network ]]; then
            network="$argument"
        elif [[ "$argument" == --network=* ]]; then
            network="${argument#--network=}"
        fi
        previous="$argument"
    done
fi
export PODMAN_AFFINITY_NETWORK="$network"

if [[ "$(id -u)" == 0 ]]; then
    if [[ "$network" != none ]]; then
        mkdir -p /dev/net
        if [[ ! -e /dev/net/tun ]]; then
            mknod /dev/net/tun c 10 200
        fi
        if [[ ! -c /dev/net/tun ]]; then
            echo "Expected a TUN character device at /dev/net/tun" >&2
            exit 1
        fi
        chown podman:podman /dev/net/tun
        chmod 600 /dev/net/tun
    fi
    install -d -m 700 -o podman -g podman "${XDG_RUNTIME_DIR:?}"
    exec setpriv --reuid=podman --regid=podman --init-groups "$0" "$@"
fi

mkdir -p "${XDG_RUNTIME_DIR:?}"
chmod 700 "$XDG_RUNTIME_DIR"

if [[ $# == 0 ]]; then
    set -- literegistry podman
elif [[ "$1" == -* ]]; then
    set -- literegistry podman "$@"
fi
exec "$@"
