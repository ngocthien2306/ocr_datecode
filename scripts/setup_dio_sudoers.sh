#!/bin/bash
#
# Setup sudoers to allow dio_in and dio_out commands without password
# This is required for camera producer to read DI in hardware trigger mode
#

echo "Setting up sudoers for Digital I/O commands..."

# Create sudoers file for dio commands
SUDOERS_FILE="/etc/sudoers.d/99-dio-nopasswd"

# Create the sudoers rule
cat << 'EOF' | sudo tee "$SUDOERS_FILE" > /dev/null
# Allow user 'demo' to run dio_in and dio_out without password
demo ALL=(ALL) NOPASSWD: /usr/sbin/dio_in, /usr/sbin/dio_out
EOF

# Set correct permissions (sudoers files must be 0440)
sudo chmod 0440 "$SUDOERS_FILE"

echo "✅ Sudoers configuration created: $SUDOERS_FILE"
echo ""
echo "Testing dio_in without password:"
sudo dio_in 0

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! You can now run 'sudo dio_in' and 'sudo dio_out' without password."
else
    echo ""
    echo "❌ ERROR: Configuration may have failed. Check sudo permissions."
    exit 1
fi

echo ""
echo "Note: Camera producer will now be able to read DI without password prompts."
