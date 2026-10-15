#!/usr/bin/env bash
# =============================================================================
# env_setup.sh
# Full environment setup for Code-Based Scalable coSNARKs benchmarks.
# Tested on Ubuntu 22.04 LTS and Ubuntu 24.04 LTS, x86-64.
# Target Azure VM: Standard_D4als_v6 (AMD EPYC Genoa, 4 vCPU) or larger.
#
# Usage:
#   chmod +x env_setup.sh
#   ./env_setup.sh              # full setup + build
#   ./env_setup.sh --test        # full setup + build + smoke test
#   ./env_setup.sh --no-build   # skip build step
#   ./env_setup.sh --no-test    # skip smoke test (default)
#
# See PREREQUISITES.md for full details.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_BUILD=false
NO_TEST=true

for arg in "$@"; do
    case "$arg" in
        --no-build) NO_BUILD=true ;;
        --test)  NO_TEST=false  ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
print_header() { echo; echo "============================================================"; echo " $1"; echo "============================================================"; }
check_cpu_features() {
    local missing=()
    grep -q "avx2" /proc/cpuinfo || missing+=("avx2")
    grep -q "sha_ni\|sha-ni" /proc/cpuinfo || missing+=("sha_ni")
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "  [WARNING] Missing CPU features: ${missing[*]}"
        echo "  The binary uses -mavx -msha flags. It may crash at runtime."
        echo "  Verify your VM/CPU supports AVX2 and SHA-NI before continuing."
    else
        echo "  [OK] CPU features: avx2 and sha_ni present."
    fi
}
# ─────────────────────────────────────────────────────────────────────────────

print_header "CPU feature check"
check_cpu_features

print_header "Step 1: System packages (requires sudo)"
sudo apt-get update -qq

# libomp-dev is versioned on Ubuntu 24 (libomp-XX-dev); pick whatever is available.
LIBOMP_PKG="libomp-dev"
if ! apt-cache show libomp-dev &>/dev/null 2>&1; then
    LIBOMP_PKG=$(apt-cache search 'libomp-[0-9]+-dev' 2>/dev/null | awk '{print $1}' | sort -V | tail -1)
    LIBOMP_PKG=${LIBOMP_PKG:-"libomp-dev"}
fi
echo "  -> OpenMP dev package: $LIBOMP_PKG"

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libgomp1 \
    "$LIBOMP_PKG" \
    iproute2 \
    python3 \
    python3-pip \
    python3-venv \
    openssh-client

print_header "Step 2: Python virtual environment + packages"
VENV_DIR="$REPO_DIR/.venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "  -> Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "  -> Reusing existing venv at $VENV_DIR"
fi

# Activate for the remainder of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install "paramiko==4.0.0" "tcconfig==0.30.1"

# Write a persistent activation line to ~/.bashrc so future SSH sessions
# (and the benchmark driver) use the venv automatically.
ACTIVATE_LINE="source \"$VENV_DIR/bin/activate\""
if ! grep -qF "$VENV_DIR/bin/activate" "$HOME/.bashrc" 2>/dev/null; then
    echo "$ACTIVATE_LINE" >> "$HOME/.bashrc"
    echo "  -> Added venv activation to ~/.bashrc"
fi

# Grant tc CAP_NET_ADMIN via a sudoers rule so that the Python benchmark
# driver can call 'sudo tcdel / sudo tcset' without a password prompt.
TC_BIN=$(which tc 2>/dev/null || echo "/sbin/tc")
SUDOERS_FILE="/etc/sudoers.d/cosnarks-tc"
if [[ ! -f "$SUDOERS_FILE" ]]; then
    echo "  -> Adding sudoers NOPASSWD rule for $TC_BIN"
    echo "$USER ALL=(ALL) NOPASSWD: $TC_BIN" | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 440 "$SUDOERS_FILE"
fi

print_header "Step 3: Verify installed tools"
echo -n "  g++:     "; g++     --version | head -1
echo -n "  cmake:   "; cmake   --version | head -1
echo -n "  mpiexec: "; mpiexec --version 2>&1 | head -1
echo -n "  python3: "; python3 --version
echo -n "  pip:     "; pip     --version | head -1
echo -n "  venv:    "; echo "$VENV_DIR"
if python3 -c "import paramiko, tcconfig" 2>/dev/null; then
    echo "  paramiko + tcconfig: OK"
else
    echo "  [ERROR] paramiko or tcconfig not importable inside venv."
fi
if command -v tcset &>/dev/null; then
    echo -n "  tcset:   "; tcset --version 2>/dev/null || echo "(installed)"
else
    echo "  tcset:   NOT FOUND inside venv — check pip install output above"
fi
echo -n "  tc:      "; tc -V 2>&1 | head -1

if [[ "$NO_BUILD" == "true" ]]; then
    echo; echo "[INFO] Skipping build step (--no-build specified)."
else
    print_header "Step 4: Build the project (Release mode)"
    cd "$REPO_DIR"
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    # Build only the pigeon target inside src/
    cd src
    make -j"$(nproc)"
    BINARY="$REPO_DIR/build/src/pigeon"
    if [[ -x "$BINARY" ]]; then
        echo "  -> Binary built successfully: $BINARY"
    else
        echo "  [ERROR] Build failed — $BINARY not found."
        exit 1
    fi
    cd "$REPO_DIR"
fi

if [[ "$NO_TEST" == "true" ]]; then
    echo; echo "[INFO] Skipping smoke test (--no-test specified)."
else
    print_header "Step 5: Smoke test (4 MPI ranks, 2^10 gates, TCP loopback)"
    cd "$REPO_DIR"
    BINARY="$REPO_DIR/build/src/pigeon"
    if [[ ! -x "$BINARY" ]]; then
        echo "  [SKIP] Binary not found. Build first, then re-run with --no-test removed."
    else
        mpiexec -n 4 \
            --mca pml ob1 \
            --mca btl tcp,self \
            --mca btl_tcp_if_include lo \
            --oversubscribe \
            "$BINARY" 0 10 1
        echo "  -> Smoke test PASSED."
    fi
fi

cat <<EOF

============================================================
 Environment setup COMPLETE
============================================================
 Python venv: $VENV_DIR
 Activate with: source $VENV_DIR/bin/activate
   (already added to ~/.bashrc — auto-active in new shells)

 Useful next steps:
   # Single-node sweep (shell-based)
   ./benchmark_suite.sh

   # Network-delay benchmark — single node, loopback interface
   source $VENV_DIR/bin/activate   # if not already active
   python3 network_simulation/driver.py --task sweep --iface lo

   # Reset latency rules at any time
   python3 network_simulation/driver.py --task reset_network --iface lo

 See PREREQUISITES.md for full documentation.
============================================================
EOF
