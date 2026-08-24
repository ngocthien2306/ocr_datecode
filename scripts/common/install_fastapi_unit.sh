#!/bin/bash
# Shared installer for the two standalone FastAPI side-services
# (ocr_service :8002, anomaly_service :8001).
#
# Sourced by scripts/setup_ocr_training_service.sh and
# scripts/setup_anomaly_service.sh. Neither is part of
# setup_systemd_services.sh (that one recreates ocr-backend + ocr-all and
# restarts the whole stack, camera check included) nor of start_services.sh
# (a camera restart must not take a training run down).
#
# Caller sets, then calls install_unit or uninstall_unit:
#   SVC_NAME     systemd unit name without .service   e.g. ocr-training
#   SVC_DESC     Description=
#   SVC_DIR      absolute WorkingDirectory
#   SVC_ENV      conda env name
#   SVC_PORT     port to bind and health-check
#   SVC_LOG      log file basename                    e.g. ocr_service.log
#   SVC_IMPORTS  python modules to import as a smoke test
#   SVC_NOTES    extra comment block for the unit (may be empty)
# Optional:
#   svc_extra_validate()   run after the standard checks; warn, don't exit,
#                          for anything that only blocks part of the service

# Resolve the SERVICE user, not the effective one. These scripts sudo for the
# few root steps, so they are meant to run WITHOUT sudo — but `sudo bash ...` is
# the reflex, and under sudo $HOME/whoami become root's, which would point
# everything at /root/Source/ocr_datecode. SUDO_USER gives the real invoker
# back; getent reads the passwd entry rather than trusting $HOME.
_resolve_user() {
    USER_NAME="${SUDO_USER:-$(whoami)}"
    USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
    if [ -z "$USER_HOME" ] || [ ! -d "$USER_HOME" ]; then
        echo "❌ Could not resolve home directory for user '${USER_NAME}'"
        exit 1
    fi
    PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
    LOG_DIR="${PROJECT_DIR}/logs"
    CONDA_SH="${USER_HOME}/miniconda3/etc/profile.d/conda.sh"
    ENV_PY="${USER_HOME}/miniconda3/envs/${SVC_ENV}/bin/python"
    UNIT="/etc/systemd/system/${SVC_NAME}.service"
}

_banner() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " ${SVC_NAME}.service — $1"
    echo " User    : $USER_NAME"
    echo " Dir     : $SVC_DIR"
    echo " Env     : $SVC_ENV        Port: $SVC_PORT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

uninstall_unit() {
    _resolve_user
    _banner uninstall
    sudo systemctl stop    "${SVC_NAME}.service" 2>/dev/null || true
    sudo systemctl disable "${SVC_NAME}.service" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl reset-failed "${SVC_NAME}.service" 2>/dev/null || true
    sudo systemctl daemon-reload
    echo "✅ Removed ${SVC_NAME}.service (nothing else touched)"
}

install_unit() {
    _resolve_user
    _banner install

    # ── Validate BEFORE writing anything ─────────────────────────────────────
    echo "Validating..."
    if [ ! -x "$ENV_PY" ]; then
        echo "❌ conda env '${SVC_ENV}' not found at ${ENV_PY}"
        exit 1
    fi
    [ -f "$CONDA_SH" ]      || { echo "❌ conda.sh not found at ${CONDA_SH}"; exit 1; }
    [ -d "${SVC_DIR}/app" ] || { echo "❌ ${SVC_DIR}/app missing — wrong project dir?"; exit 1; }
    if [ ! -f "${SVC_DIR}/.env" ]; then
        echo "❌ ${SVC_DIR}/.env missing — cp .env.sample .env, and set SECRET_KEY to"
        echo "   MATCH backend/.env exactly or every endpoint answers 401."
        exit 1
    fi
    # Import for real: these packages live in the env, not the base interpreter,
    # and a missing one only shows up as a crashloop under systemd otherwise.
    "$ENV_PY" -c "import ${SVC_IMPORTS}" 2>/dev/null \
        || { echo "❌ env '${SVC_ENV}' cannot import: ${SVC_IMPORTS}"; exit 1; }
    echo "   ✅ env, app/, .env and imports all present"
    if declare -f svc_extra_validate >/dev/null; then svc_extra_validate; fi
    echo ""

    mkdir -p "$LOG_DIR"

    echo "Writing ${UNIT}..."
    sudo tee "$UNIT" > /dev/null << EOF
[Unit]
Description=${SVC_DESC}
After=network.target mongod.service
Wants=mongod.service

[Service]
User=${USER_NAME}
Group=${USER_NAME}
WorkingDirectory=${SVC_DIR}
# uvicorn directly, NOT \`python -m app.main\`: that entry point passes
# reload=settings.DEBUG and .env ships DEBUG=True, so under systemd a reload
# watcher would restart the process on any file touch — killing a training run
# that was minutes in.
#
# WorkingDirectory matters more than it looks: the repo ROOT also has an
# app/main.py (an unrelated PyQt5 tool), so a wrong cwd imports the wrong app.
ExecStart=/bin/bash -lc 'source ${CONDA_SH} && conda activate ${SVC_ENV} && exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${SVC_PORT}'
Restart=always
RestartSec=5
# Generous: stopping has to let a CUDA/TensorRT context unwind.
TimeoutStopSec=60
#
# PrivateTmp is intentionally absent. The cross-service GPU lock lives at
# /tmp/ocr_datecode_gpu.lock and has to be the SAME file every participant sees;
# PrivateTmp=yes would give this unit a private /tmp and silently disable the
# mutual exclusion.
#
# StartLimitIntervalSec/StartLimitBurst are also absent. ocr-backend.service has
# them in [Service], where systemd ignores them (systemd-analyze verify:
# "Unknown key name"), so they have never limited anything there. Moving them to
# [Unit] would make them real and let this unit give up after N restarts — for an
# operator-facing API, silently staying down is worse than retrying.
${SVC_NOTES}
StandardOutput=append:${LOG_DIR}/${SVC_LOG}
StandardError=append:${LOG_DIR}/${SVC_LOG}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "${SVC_NAME}.service" >/dev/null
    sudo systemctl restart "${SVC_NAME}.service"
    echo ""

    # Importing torch/anomalib/cv2 in these envs is not instant.
    for _ in $(seq 1 25); do
        curl -sf -o /dev/null "http://127.0.0.1:${SVC_PORT}/health" && break
        sleep 1
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if curl -sf -o /dev/null "http://127.0.0.1:${SVC_PORT}/health"; then
        echo "  ✅ ${SVC_NAME}.service — healthy on :${SVC_PORT}"
    else
        echo "  ❌ not answering /health"
        echo "     sudo journalctl -u ${SVC_NAME} -n 50 --no-pager"
        echo "     tail -50 ${LOG_DIR}/${SVC_LOG}"
    fi
    echo ""
    echo "  sudo systemctl restart ${SVC_NAME}"
    echo "  sudo journalctl -u ${SVC_NAME} -f"
    echo "  tail -f ${LOG_DIR}/${SVC_LOG}"
    echo ""
    echo "  bash $(basename "$0") --uninstall   # remove"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
