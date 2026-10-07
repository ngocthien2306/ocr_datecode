#!/bin/bash
# Install ONLY the PaddleOCR Training API unit (ocr_paddle_service, :8003).
#
#   bash scripts/setup_ocr_paddle_service.sh              # install + start
#   bash scripts/setup_ocr_paddle_service.sh --uninstall  # remove it
#   bash scripts/setup_ocr_paddle_service.sh --download PP-OCRv6_small_rec  # pretrained weights
#
# Clone of ocr-training on its own conda env (paddle_train) so paddlepaddle-gpu
# never shares a site-packages with torch/OpenOCR. See
# scripts/common/install_fastapi_unit.sh for why this is not part of
# setup_systemd_services.sh / start_services.sh.
set -e
source "$(dirname "$0")/common/install_fastapi_unit.sh"

SVC_NAME=ocr-paddle-training
SVC_DESC="OCR Datecode PaddleOCR Training API (:8003)"
SVC_DIR="$(cd "$(dirname "$0")/.." && pwd)/ocr_paddle_service"
SVC_ENV=paddle_train
SVC_PORT=8003
SVC_LOG=ocr_paddle_service.log
SVC_IMPORTS="uvicorn, fastapi, motor, paddle"
# Same reasoning as ocr-training: the trainer is a Popen child that owns its
# VRAM, so the default control-group kill must take it down with the unit.
SVC_NOTES='#
# No KillMode override: training runs PaddleOCR tools/train.py as a separate
# process that owns its own VRAM. Killing only the parent would leave it alive
# with the GPU flock released, and the next run would start a second trainer.
# The default control-group kill takes the trainer down with the unit.'

svc_extra_validate() {
    [ -f "${SVC_DIR}/PaddleOCR/tools/train.py" ] \
        || echo "   ⚠️  ${SVC_DIR}/PaddleOCR missing — TRAINING will fail (git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git)"
    ls "${SVC_DIR}"/weights/paddle_pretrained/*.pdparams >/dev/null 2>&1 \
        || echo "   ⚠️  no *_pretrained.pdparams — run: bash $0 --download PP-OCRv6_small_rec"
    # Engines are built + scored in the env ai_services runs in (same TensorRT).
    [ -d "${HOME}/miniconda3/envs/vision" ] \
        || echo "   ⚠️  conda env 'vision' missing — TensorRT export will fail (TRT_CONDA_ENV in .env)"
}

# Official PP-OCR rec weights to fine-tune from (gitignored, 60-130 MB each):
#   bash scripts/setup_ocr_paddle_service.sh --download PP-OCRv6_small_rec [PP-OCRv5_mobile_rec ...]
download_pretrained() {
    local dir="${SVC_DIR}/weights/paddle_pretrained"
    mkdir -p "$dir"
    for arch in "$@"; do
        echo "⬇️  ${arch}"
        wget -q --show-progress -c -O "${dir}/${arch}_pretrained.pdparams" \
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/${arch}_pretrained.pdparams"
    done
}

case "${1:-}" in
    --uninstall) uninstall_unit ;;
    --download)  shift; download_pretrained "${@:-PP-OCRv6_small_rec}" ;;
    *)           install_unit ;;
esac
