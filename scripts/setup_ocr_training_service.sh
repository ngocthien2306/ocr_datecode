#!/bin/bash
# Install ONLY the OCR Training API unit (ocr_service, :8002).
#
#   bash scripts/setup_ocr_training_service.sh              # install + start
#   bash scripts/setup_ocr_training_service.sh --uninstall  # remove it
#
# Nothing else is touched — see scripts/common/install_fastapi_unit.sh for why this
# is separate from setup_systemd_services.sh and start_services.sh.
set -e
source "$(dirname "$0")/common/install_fastapi_unit.sh"

SVC_NAME=ocr-training
SVC_DESC="OCR Datecode OCR Training API (:8002)"
SVC_DIR="$(cd "$(dirname "$0")/.." && pwd)/ocr_service"
SVC_ENV=ocr_train
SVC_PORT=8002
SVC_LOG=ocr_service.log
SVC_IMPORTS="uvicorn, fastapi, motor"
# Training here spawns OpenOCR's train_rec.py as a SEPARATE process, which is
# why the kill mode is worth a paragraph.
SVC_NOTES='#
# No KillMode override, unlike ocr-backend which needs '"'"'mixed'"'"'. Training spawns
# train_rec.py with a plain subprocess.Popen: on Linux that child is not tied to
# its parent'"'"'s lifetime, and its CUDA memory belongs to IT — VRAM is owned per
# process and released only when that process exits, and PyTorch'"'"'s caching
# allocator holds its reserved pool even while idle. Measured: kill the parent and
# the child keeps running while the GPU flock IS released (Popen defaults to
# close_fds=True, so the child never inherited the lock fd). Lock free plus orphan
# alive is the dangerous combination — the next run sees a free GPU and starts a
# SECOND trainer beside it. The default control-group kill takes the trainer down
# with the unit, which is what we want.'

svc_extra_validate() {
    # Not fatal: the label-review half of the studio works without OpenOCR or the
    # base checkpoints, and refusing to install would block that too.
    [ -d "${SVC_DIR}/OpenOCR" ] \
        || echo "   ⚠️  ${SVC_DIR}/OpenOCR missing — TRAINING will fail (see ocr_service/README.md)"
    ls "${SVC_DIR}"/weights/base/*.pth >/dev/null 2>&1 \
        || echo "   ⚠️  no base checkpoints in ${SVC_DIR}/weights/base — training will fail"
}

if [ "${1:-}" = "--uninstall" ]; then uninstall_unit; else install_unit; fi
