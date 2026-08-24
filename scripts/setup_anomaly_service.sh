#!/bin/bash
# Install ONLY the Anomaly Training API unit (anomaly_service, :8001).
#
#   bash scripts/setup_anomaly_service.sh              # install + start
#   bash scripts/setup_anomaly_service.sh --uninstall  # remove it
#
# Nothing else is touched — see scripts/common/install_fastapi_unit.sh for why this
# is separate from setup_systemd_services.sh and start_services.sh.
#
# Until now this service was started by hand, so it did not come back after a
# reboot and nothing restarted it if it died mid-session.
set -e
source "$(dirname "$0")/common/install_fastapi_unit.sh"

SVC_NAME=ocr-anomaly
SVC_DESC="OCR Datecode Anomaly Training API (:8001)"
SVC_DIR="$(cd "$(dirname "$0")/.." && pwd)/anomaly_service"
SVC_ENV=anomaly_service
SVC_PORT=8001
SVC_LOG=anomaly_service.log
SVC_IMPORTS="uvicorn, fastapi, motor, anomalib"
# Unlike ocr_service, anomaly training runs IN-PROCESS (run_in_executor, no
# subprocess), so there is no orphan trainer to reason about — but the flip side
# matters more, hence the note.
SVC_NOTES='#
# Anomaly training runs IN-PROCESS (run_in_executor, not a subprocess), so
# stopping or restarting this unit aborts a run in progress immediately — there
# is no orphan trainer left behind, and equally no chance for it to finish. That
# is also exactly why the reload watcher had to go: the service has been running
# with --reload, where a single file touch killed whatever was training.
#
# It does NOT take the /tmp/ocr_datecode_gpu.lock that ocr_service uses, so an
# anomaly run and an OCR run can still collide on the GPU. Fixing that means
# wrapping the training call in ocr_service'"'"'s gpu_lock context manager pointed at
# the same file.'

svc_extra_validate() {
    # Training fits PatchCore/Padim on normal images only, but with no dataset at
    # all there is nothing to do — still not a reason to refuse the install.
    [ -d "${SVC_DIR}/data/projects" ] \
        || echo "   ⚠️  ${SVC_DIR}/data/projects missing — no datasets imported yet"
}

if [ "${1:-}" = "--uninstall" ]; then uninstall_unit; else install_unit; fi
