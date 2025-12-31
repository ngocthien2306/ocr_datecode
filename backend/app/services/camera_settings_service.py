"""
Camera Settings Service
Service to manage camera runtime settings via JSON files
"""
import json
import os
import logging
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

# Setup logging to file
LOGS_DIR = Path("/home/demo/Source/ocr_datecode/backend/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure file handler for settings service
file_handler = logging.FileHandler(LOGS_DIR / 'camera_settings.log', mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class CameraSettingsService:
    """Service to manage camera settings files"""

    def __init__(self, settings_dir: str = "camera_settings"):
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 [INIT] Camera settings directory: {self.settings_dir}")

    def _get_settings_file(self, serial_number: str) -> Path:
        """Get settings file path for a camera"""
        return self.settings_dir / f"{serial_number}.json"

    def get_settings(self, serial_number: str) -> Optional[Dict]:
        """
        Get current settings for a camera

        Args:
            serial_number: Camera serial number

        Returns:
            Dict with settings or None if not found
        """
        settings_file = self._get_settings_file(serial_number)

        if not settings_file.exists():
            logger.info(f"⚠️  [GET SETTINGS] No settings file found for camera {serial_number}")
            return None

        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                logger.info(f"✅ [GET SETTINGS] Camera {serial_number}: {settings}")
                return settings
        except Exception as e:
            logger.error(f"❌ [GET SETTINGS ERROR] Camera {serial_number}: {e}")
            return None

    def update_settings(
        self,
        serial_number: str,
        exposure_time: Optional[int] = None,
        gain: Optional[float] = None,
        trigger_config: Optional[Dict] = None
    ) -> Dict:
        """
        Update settings for a camera

        Args:
            serial_number: Camera serial number
            exposure_time: Exposure time in microseconds (optional)
            gain: Camera gain value (optional)
            trigger_config: Trigger configuration dict (optional)

        Returns:
            Dict with updated settings
        """
        settings_file = self._get_settings_file(serial_number)

        # Read existing settings or create new
        old_settings = None
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    old_settings = json.load(f)
                    settings = old_settings.copy()
            except:
                settings = {}
        else:
            settings = {}
            logger.info(f"📝 [NEW SETTINGS] Creating new settings file for camera {serial_number}")

        # Track changes
        changes = []

        # Update settings
        if exposure_time is not None:
            old_val = settings.get('exposure_time', 'None')
            settings['exposure_time'] = exposure_time
            changes.append(f"exposure_time: {old_val} → {exposure_time}")

        if gain is not None:
            old_val = settings.get('gain', 'None')
            settings['gain'] = gain
            changes.append(f"gain: {old_val} → {gain}")

        if trigger_config is not None:
            old_val = settings.get('trigger_config', {})
            settings['trigger_config'] = trigger_config
            trigger_mode = trigger_config.get('mode', 'continuous')
            changes.append(f"trigger_mode: {old_val.get('mode', 'N/A') if isinstance(old_val, dict) else 'N/A'} → {trigger_mode}")

        # Write settings to file
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            if changes:
                logger.info(f"✅ [UPDATE SETTINGS] Camera {serial_number}: {', '.join(changes)}")
            logger.info(f"💾 [SAVED TO FILE] {settings_file}")
            logger.info(f"📊 [NEW STATE] Camera {serial_number}: {settings}")

            return settings
        except Exception as e:
            logger.error(f"❌ [UPDATE SETTINGS ERROR] Camera {serial_number}: {e}")
            raise

    def delete_settings(self, serial_number: str) -> bool:
        """
        Delete settings file for a camera

        Args:
            serial_number: Camera serial number

        Returns:
            True if deleted, False if not found
        """
        settings_file = self._get_settings_file(serial_number)

        if settings_file.exists():
            try:
                settings_file.unlink()
                logger.info(f"🗑️  [DELETE SETTINGS] Camera {serial_number}: Settings file deleted")
                return True
            except Exception as e:
                logger.error(f"❌ [DELETE SETTINGS ERROR] Camera {serial_number}: {e}")
                return False

        logger.info(f"⚠️  [DELETE SETTINGS] Camera {serial_number}: No settings file to delete")
        return False


# Singleton instance
camera_settings_service = CameraSettingsService()
