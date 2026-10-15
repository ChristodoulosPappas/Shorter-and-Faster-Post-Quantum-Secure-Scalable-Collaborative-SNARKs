#!/usr/bin/env bash
# =============================================================================
# backup_leader_results.sh
# Connects to the Leader VM, archives all benchmark results and logs, 
# and downloads the archive to the local backups/ directory.
# =============================================================================
set -euo pipefail

# Configuration
NUM_VMS=${1:-32} # Default to 32 VMs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="$SCRIPT_DIR/scripts/ips_${NUM_VMS}.txt"
BACKUP_DIR="$SCRIPT_DIR/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="cocoSNARK_leader_backup_$TIMESTAMP.tar.gz"
REMOTE_USER="azureuser"
REMOTE_REPO="~/sPQcocoSNARK" # Standard path on VM

echo "============================================================"
echo " Starting Remote Leader Backup ($NUM_VMS VM configuration)"
echo " Time: $TIMESTAMP"
echo "============================================================"

# 1. Resolve Leader IP
if [[ ! -f "$IPS_FILE" ]]; then
    echo "[ERROR] IPs file not found: $IPS_FILE"
    echo "Please specify the correct VM count: $0 [VM_COUNT]"
    exit 1
fi

LEADER_IP=$(grep "leader" "$IPS_FILE" | awk '{print $2}')
if [[ -z "$LEADER_IP" ]]; then
    echo "[ERROR] Could not find leader IP in $IPS_FILE"
    exit 1
fi

echo " -> Leader VM Public IP: $LEADER_IP"
mkdir -p "$BACKUP_DIR"

# 2. Remote Archiving Command
# This runs on the leader VM to collect all logs and benchmark_results directories.
# It prefixes each item with its modification timestamp to prevent name collisions.
REMOTE_CMD="
    cd $REMOTE_REPO && \
    mkdir -p backups_temp && \
    
    echo '  -> Processing logs...'
    find . -maxdepth 2 -name '*.log' | while read f; do
        ts=\$(date -r \"\$f\" +\"%Y%m%d_%H%M%S\")
        fname=\$(basename \"\$f\")
        cp \"\$f\" \"backups_temp/\${ts}_\${fname}\"
    done
    
    echo '  -> Processing result directories...'
    for d in benchmark_results_*; do
        [ -d \"\$d\" ] || continue
        ts=\$(date -r \"\$d\" +\"%Y%m%d_%H%M%S\")
        cp -r \"\$d\" \"backups_temp/\${ts}_\${d}\"
    done
    
    if [ \"\$(ls -A backups_temp)\" ]; then
        tar -czf $ARCHIVE_NAME -C backups_temp . && \
        rm -rf backups_temp && \
        echo 'ARCHIVE_READY'
    else
        rm -rf backups_temp && \
        echo 'NO_RESULTS'
    fi
"

echo "3. Executing Remote Archive on Leader..."
RESULT=$(ssh -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$LEADER_IP" "$REMOTE_CMD")

if [[ "$RESULT" == *"ARCHIVE_READY"* ]]; then
    echo "4. Downloading Archive to Local Storage..."
    scp "$REMOTE_USER@$LEADER_IP:$REMOTE_REPO/$ARCHIVE_NAME" "$BACKUP_DIR/"
    
    # Cleanup remote archive
    ssh "$REMOTE_USER@$LEADER_IP" "rm $REMOTE_REPO/$ARCHIVE_NAME"
    
    echo "============================================================"
    echo " Backup Successful!"
    echo " Location: $BACKUP_DIR/$ARCHIVE_NAME"
    echo " Size: $(du -h "$BACKUP_DIR/$ARCHIVE_NAME" | cut -f1)"
    echo "============================================================"
elif [[ "$RESULT" == *"NO_RESULTS"* ]]; then
    echo "[WARNING] No logs or benchmark_results directories found on the Leader VM."
else
    echo "[ERROR] Remote backup failed."
    echo "$RESULT"
    exit 1
fi
