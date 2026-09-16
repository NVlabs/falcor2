#!/usr/bin/env bash

# Install only the Debian/Ubuntu user-space packages required to configure and
# build Falcor2. Machine provisioning concerns such as kernel headers, DKMS,
# driver installation, and CI-only coverage tools deliberately do not belong
# here.

set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-query >/dev/null 2>&1; then
    echo "Falcor2 prerequisites currently support Debian/Ubuntu systems only." >&2
    exit 1
fi

packages=(
    # This provides the distro-default gcc/g++ commands expected by the
    # linux-gcc CMake preset without pinning Falcor2 to a compiler version.
    build-essential
    ca-certificates
    clang
    cmake
    curl
    git
    git-lfs
    libgtk-3-dev
    ninja-build
    python3
    python3-pip
    python3-venv
)

# Avoid an unnecessary apt update/install on machines that are already ready.
missing_packages=()
for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q '^install ok installed$'; then
        missing_packages+=("${package}")
    fi
done

if ((${#missing_packages[@]} == 0)); then
    exit 0
fi

echo "Installing missing Falcor2 host prerequisites: ${missing_packages[*]}"

# Use apt directly when already root; otherwise require sudo rather than
# failing later with a less useful package-manager permissions error.
if ((EUID == 0)); then
    apt-get update
    env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_packages[@]}"
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Installing prerequisites requires root privileges or sudo." >&2
        exit 1
    fi
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_packages[@]}"
fi

echo "Falcor2 host prerequisites installed."
