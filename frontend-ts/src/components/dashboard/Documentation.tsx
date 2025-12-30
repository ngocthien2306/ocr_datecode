import { useState } from 'react';
import '@/styles/Documentation.css';

type DocSection =
  | 'overview'
  | 'dashboard'
  | 'users'
  | 'recipes'
  | 'cameras'
  | 'historical'
  | 'settings'
  | 'logs'
  | 'permissions';

const Documentation: React.FC = () => {
  const [activeSection, setActiveSection] = useState<DocSection>('overview');

  const renderContent = () => {
    switch (activeSection) {
      case 'overview':
        return (
          <div className="doc-content">
            <h1>📚 System Overview</h1>
            <p className="doc-intro">
              Welcome to the <strong>Suntech Automation OCR Date Code Inspection System</strong>.
              This comprehensive platform provides real-time camera monitoring, recipe management,
              and automated quality control for production lines.
            </p>

            <div className="doc-section">
              <h2>🎯 Key Features</h2>
              <ul className="feature-list">
                <li>
                  <strong>Real-time Camera Monitoring</strong>
                  <p>Monitor multiple cameras simultaneously with live feed and status updates</p>
                </li>
                <li>
                  <strong>Recipe Management</strong>
                  <p>Create, edit, and load OCR inspection recipes for different products</p>
                </li>
                <li>
                  <strong>User Access Control</strong>
                  <p>Role-based permissions (Admin, Supervisor, Operator) with granular access control</p>
                </li>
                <li>
                  <strong>Historical Data Analysis</strong>
                  <p>Track inspection results, view statistics, and export reports</p>
                </li>
                <li>
                  <strong>System Configuration</strong>
                  <p>Customize loading animations, templates, and system preferences</p>
                </li>
                <li>
                  <strong>Activity Logging</strong>
                  <p>Comprehensive audit trail of all system actions and user activities</p>
                </li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🚀 Getting Started</h2>
              <ol className="getting-started-list">
                <li>
                  <strong>Login</strong>
                  <p>Enter your credentials on the login page. Contact your administrator if you don't have an account.</p>
                </li>
                <li>
                  <strong>Dashboard Overview</strong>
                  <p>After login, you'll see the main dashboard with camera status, production metrics, and recent activities.</p>
                </li>
                <li>
                  <strong>Navigate Sections</strong>
                  <p>Use the sidebar menu to access different sections based on your permissions.</p>
                </li>
                <li>
                  <strong>Load Recipe</strong>
                  <p>Go to Recipes page, select a recipe, and click "Load" to activate it for production.</p>
                </li>
                <li>
                  <strong>Monitor Production</strong>
                  <p>View real-time camera feeds and inspection results on the Dashboard and Historical pages.</p>
                </li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>💡 Quick Tips</h2>
              <div className="tips-grid">
                <div className="tip-card">
                  <div className="tip-icon">⌨️</div>
                  <h3>Keyboard Shortcuts</h3>
                  <p>Use arrow keys to navigate between pages quickly</p>
                </div>
                <div className="tip-card">
                  <div className="tip-icon">🎨</div>
                  <h3>Dark Mode</h3>
                  <p>Toggle dark/light mode in the header for comfortable viewing</p>
                </div>
                <div className="tip-card">
                  <div className="tip-icon">🔔</div>
                  <h3>Notifications</h3>
                  <p>Watch for toast notifications for action confirmations and errors</p>
                </div>
                <div className="tip-card">
                  <div className="tip-icon">📊</div>
                  <h3>Export Data</h3>
                  <p>Export historical data and logs to Excel for offline analysis</p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'dashboard':
        return (
          <div className="doc-content">
            <h1>📊 Dashboard</h1>
            <p className="doc-intro">
              The Dashboard provides a comprehensive overview of your production system with real-time metrics and camera previews.
            </p>

            <div className="doc-section">
              <h2>📈 Summary Cards</h2>
              <div className="cards-explanation">
                <div className="card-explain">
                  <h3>🎥 Total Cameras</h3>
                  <p>Displays the total number of cameras in the system and their online status.</p>
                  <ul>
                    <li><span className="badge green">All online</span> - All cameras are connected</li>
                    <li><span className="badge yellow">X online</span> - Some cameras offline</li>
                  </ul>
                </div>
                <div className="card-explain">
                  <h3>📦 Products Today</h3>
                  <p>Shows the total number of products inspected today with percentage change from yesterday.</p>
                </div>
                <div className="card-explain">
                  <h3>⏱️ Avg Processing Time</h3>
                  <p>Average time taken to process and inspect each product.</p>
                </div>
                <div className="card-explain">
                  <h3>✅ Success Rate</h3>
                  <p>Percentage of products that passed inspection successfully.</p>
                </div>
              </div>
            </div>

            <div className="doc-section">
              <h2>🎥 Camera Previews</h2>
              <p>View live feed from up to 3 cameras simultaneously:</p>
              <ul>
                <li><span className="badge green">LIVE</span> badge indicates active streaming</li>
                <li>Camera ID and location are displayed in the header</li>
                <li>Click on a camera preview to view full-screen feed (if available)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📋 Recent Loads & Members</h2>
              <div className="two-columns">
                <div>
                  <h3>Recent Recipe Loads</h3>
                  <p>Shows the last 5 recipes loaded into the system:</p>
                  <ul>
                    <li>Recipe name and product code</li>
                    <li>User who loaded it</li>
                    <li>Load timestamp</li>
                  </ul>
                </div>
                <div>
                  <h3>Recent Members</h3>
                  <p>Displays recently active users:</p>
                  <ul>
                    <li>User avatar and name</li>
                    <li>Role (Admin/Manager/Operator)</li>
                    <li>Last active time</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>Access Level:</strong> All users can view the Dashboard</p>
            </div>
          </div>
        );

      case 'users':
        return (
          <div className="doc-content">
            <h1>👥 User Management</h1>
            <p className="doc-intro">
              Manage system users, roles, and permissions. Create new accounts, edit user information, and control access levels.
            </p>

            <div className="doc-section">
              <h2>➕ Creating a New User</h2>
              <ol>
                <li>Click the <strong>"Add User"</strong> button in the top-right corner</li>
                <li>Fill in the user information:
                  <ul>
                    <li><strong>Username:</strong> Unique identifier for login (required)</li>
                    <li><strong>Full Name:</strong> Display name (required)</li>
                    <li><strong>Email:</strong> User's email address (optional)</li>
                    <li><strong>Password:</strong> Initial password (required, minimum 6 characters)</li>
                    <li><strong>Role:</strong> Select from Admin, Manager, or Operator</li>
                  </ul>
                </li>
                <li>Upload an avatar image (optional, JPG/PNG, max 5MB)</li>
                <li>Click <strong>"Create User"</strong> to save</li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>✏️ Editing User Information</h2>
              <ol>
                <li>Locate the user in the table</li>
                <li>Click the <strong>Edit</strong> button (pencil icon)</li>
                <li>Modify the information as needed</li>
                <li>Click <strong>"Update User"</strong> to save changes</li>
              </ol>
              <p className="note">
                <strong>Note:</strong> Username cannot be changed after creation. To change password, leave the password field empty if you don't want to update it.
              </p>
            </div>

            <div className="doc-section">
              <h2>🗑️ Deleting Users</h2>
              <ol>
                <li>Click the <strong>Delete</strong> button (trash icon) next to the user</li>
                <li>Confirm the deletion in the dialog</li>
              </ol>
              <p className="warning">
                <strong>⚠️ Warning:</strong> Deleting a user is permanent and cannot be undone. All user activity history will remain in the logs.
              </p>
            </div>

            <div className="doc-section">
              <h2>🔍 User Table Features</h2>
              <ul>
                <li><strong>Search:</strong> Use the search box to find users by name or username</li>
                <li><strong>Sort:</strong> Click column headers to sort by that field</li>
                <li><strong>Pagination:</strong> Navigate through pages if you have many users</li>
                <li><strong>Avatar:</strong> Hover over avatars to see full name</li>
                <li><strong>Role Badge:</strong> Color-coded badges indicate user roles</li>
              </ul>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>Access Level:</strong> Admin or Manager only</p>
              <p>Operators cannot access User Management.</p>
            </div>
          </div>
        );

      case 'recipes':
        return (
          <div className="doc-content">
            <h1>📋 Recipe Management</h1>
            <p className="doc-intro">
              Create, edit, and manage OCR inspection recipes for different products. Load recipes into the system for production use.
            </p>

            <div className="doc-section">
              <h2>📝 Creating a New Recipe</h2>
              <ol>
                <li>Click the <strong>"Add Recipe"</strong> button</li>
                <li>Fill in the recipe details:
                  <ul>
                    <li><strong>Recipe Name:</strong> Descriptive name for the recipe</li>
                    <li><strong>Product Code:</strong> Unique product identifier</li>
                    <li><strong>Description:</strong> Additional information about the product</li>
                    <li><strong>OCR Template:</strong> Select the inspection template</li>
                    <li><strong>Camera Settings:</strong> Configure camera-specific parameters</li>
                  </ul>
                </li>
                <li>Upload reference images if required</li>
                <li>Click <strong>"Create Recipe"</strong></li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>🎯 Loading a Recipe</h2>
              <p>To activate a recipe for production:</p>
              <ol>
                <li>Find the recipe in the table</li>
                <li>Click the <strong>"Load"</strong> button (clock icon)</li>
                <li>Confirm the action in the dialog</li>
                <li>Watch the loading animation - it shows the system is applying the recipe configuration</li>
                <li>Wait for the success notification</li>
              </ol>
              <p className="note">
                <strong>💡 Loading Animation:</strong> The system displays one of 6 different loading animations while applying the recipe:
                <ul>
                  <li>🔍 OCR Scanner - Text recognition scanning</li>
                  <li>📷 Camera Vision - Grid-based vision system</li>
                  <li>📊 Barcode Scanner - Laser barcode detection</li>
                  <li>🧠 Neural Network - AI processing visualization</li>
                  <li>🔲 QR Detector - QR code detection system</li>
                  <li>⚙️ Industrial Factory - Gear system with LED indicators</li>
                </ul>
                You can customize which animation appears in Settings → Recipe Load Animation.
              </p>
            </div>

            <div className="doc-section">
              <h2>✏️ Editing Recipes</h2>
              <ol>
                <li>Click the <strong>Edit</strong> button on the recipe row</li>
                <li>Modify the recipe parameters</li>
                <li>Save your changes</li>
              </ol>
              <p className="note">
                <strong>Note:</strong> Editing a recipe does not affect the currently loaded recipe. You must load it again to apply changes.
              </p>
            </div>

            <div className="doc-section">
              <h2>📥 Export & Import</h2>
              <div className="two-columns">
                <div>
                  <h3>Export Recipe</h3>
                  <p>Download recipe as JSON file:</p>
                  <ol>
                    <li>Click <strong>Export</strong> button</li>
                    <li>Save the .json file</li>
                  </ol>
                </div>
                <div>
                  <h3>Import Recipe</h3>
                  <p>Load recipe from JSON file:</p>
                  <ol>
                    <li>Click <strong>"Import Recipe"</strong></li>
                    <li>Select the .json file</li>
                    <li>Review and confirm</li>
                  </ol>
                </div>
              </div>
            </div>

            <div className="doc-section">
              <h2>📊 Recipe Statistics</h2>
              <p>At the top of the page, you'll see:</p>
              <ul>
                <li><strong>Total Recipes:</strong> Number of recipes in the system</li>
                <li><strong>Active Recipe:</strong> Currently loaded recipe (if any)</li>
                <li><strong>Last Load:</strong> When the active recipe was loaded</li>
              </ul>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>View Recipes:</strong> All users</p>
              <p><strong>Create/Edit/Delete:</strong> Admin and Manager only</p>
              <p><strong>Load Recipe:</strong> Admin, Manager, and Operator (with recipeLoad permission)</p>
            </div>
          </div>
        );

      case 'cameras':
        return (
          <div className="doc-content">
            <h1>🎥 Camera Management</h1>
            <p className="doc-intro">
              Configure and monitor cameras in the inspection system. Manage camera settings, connections, and live feeds.
            </p>

            <div className="doc-section">
              <h2>➕ Adding a Camera</h2>
              <ol>
                <li>Click <strong>"Add Camera"</strong> button</li>
                <li>Configure camera parameters:
                  <ul>
                    <li><strong>Camera ID:</strong> Unique identifier (e.g., CAM-001)</li>
                    <li><strong>Model Name:</strong> Camera model</li>
                    <li><strong>Serial Number:</strong> Hardware serial number</li>
                    <li><strong>Location:</strong> Physical location (e.g., Line 1, Station A)</li>
                    <li><strong>IP Address:</strong> Network IP address</li>
                    <li><strong>Resolution:</strong> Width × Height (e.g., 1920 × 1080)</li>
                    <li><strong>Frame Rate:</strong> FPS (frames per second)</li>
                  </ul>
                </li>
                <li>Click <strong>"Add Camera"</strong> to save</li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>🔌 Camera Status</h2>
              <p>Cameras can have the following statuses:</p>
              <div className="status-grid">
                <div className="status-item">
                  <span className="badge green">Connected</span>
                  <p>Camera is online and streaming</p>
                </div>
                <div className="status-item">
                  <span className="badge red">Disconnected</span>
                  <p>Camera is offline or unreachable</p>
                </div>
                <div className="status-item">
                  <span className="badge yellow">Active</span>
                  <p>Camera is in use for inspection</p>
                </div>
                <div className="status-item">
                  <span className="badge gray">Inactive</span>
                  <p>Camera is connected but not in use</p>
                </div>
              </div>
            </div>

            <div className="doc-section">
              <h2>⚙️ Camera Actions</h2>
              <ul>
                <li><strong>Edit:</strong> Modify camera settings and configuration</li>
                <li><strong>Delete:</strong> Remove camera from the system</li>
                <li><strong>Connect/Disconnect:</strong> Manually control camera connection</li>
                <li><strong>Activate/Deactivate:</strong> Enable/disable camera for inspection</li>
                <li><strong>View Live Feed:</strong> Open live stream preview (if available)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📸 Live Feed Preview</h2>
              <p>Click on a camera row to view its live feed:</p>
              <ul>
                <li>Full-screen or windowed view</li>
                <li>Real-time stream with minimal latency</li>
                <li>Current FPS and resolution display</li>
                <li>Snapshot capture functionality</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🔧 Troubleshooting</h2>
              <div className="troubleshooting">
                <h3>Camera Won't Connect</h3>
                <ul>
                  <li>Verify IP address is correct</li>
                  <li>Check network connectivity</li>
                  <li>Ensure camera is powered on</li>
                  <li>Check firewall settings</li>
                </ul>
                <h3>Poor Image Quality</h3>
                <ul>
                  <li>Adjust resolution settings</li>
                  <li>Clean camera lens</li>
                  <li>Check lighting conditions</li>
                  <li>Verify network bandwidth</li>
                </ul>
              </div>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>View Cameras:</strong> All users</p>
              <p><strong>Add/Edit/Delete:</strong> Admin and Manager only</p>
              <p><strong>Connect/Activate:</strong> Admin, Manager, and Operator</p>
            </div>
          </div>
        );

      case 'historical':
        return (
          <div className="doc-content">
            <h1>📊 Historical Data</h1>
            <p className="doc-intro">
              View and analyze inspection history, statistics, and results. Export data for reporting and quality control.
            </p>

            <div className="doc-section">
              <h2>🔍 Filtering Data</h2>
              <p>Use the filter controls to narrow down results:</p>
              <ul>
                <li><strong>Date Range:</strong> Select start and end dates</li>
                <li><strong>Camera:</strong> Filter by specific camera</li>
                <li><strong>Result:</strong> Pass, Fail, or All</li>
                <li><strong>Product Code:</strong> Filter by product/recipe</li>
                <li><strong>Search:</strong> Free-text search in OCR results</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📈 Statistics View</h2>
              <p>The statistics panel shows:</p>
              <div className="stats-grid">
                <div className="stat-card">
                  <h3>Total Inspections</h3>
                  <p>Number of products inspected in selected period</p>
                </div>
                <div className="stat-card">
                  <h3>Pass Rate</h3>
                  <p>Percentage of products that passed inspection</p>
                </div>
                <div className="stat-card">
                  <h3>Fail Rate</h3>
                  <p>Percentage of products that failed</p>
                </div>
                <div className="stat-card">
                  <h3>Average Time</h3>
                  <p>Mean processing time per product</p>
                </div>
              </div>
            </div>

            <div className="doc-section">
              <h2>📋 Results Table</h2>
              <p>Each row in the table shows:</p>
              <ul>
                <li><strong>Timestamp:</strong> When the inspection occurred</li>
                <li><strong>Camera:</strong> Which camera performed the inspection</li>
                <li><strong>Product Code:</strong> Recipe/product identifier</li>
                <li><strong>OCR Result:</strong> Text extracted from the image</li>
                <li><strong>Result:</strong> Pass/Fail status with confidence score</li>
                <li><strong>Processing Time:</strong> How long the inspection took</li>
                <li><strong>Image:</strong> Thumbnail preview (click to enlarge)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📥 Exporting Data</h2>
              <ol>
                <li>Apply filters to select the data you want to export</li>
                <li>Click the <strong>"Export to Excel"</strong> button</li>
                <li>Choose export options:
                  <ul>
                    <li>Current page only</li>
                    <li>All filtered results</li>
                    <li>Include images (increases file size)</li>
                  </ul>
                </li>
                <li>Download the .xlsx file</li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>📊 Charts & Graphs</h2>
              <p>Visual representations of data:</p>
              <ul>
                <li><strong>Trend Chart:</strong> Pass/fail rates over time</li>
                <li><strong>Camera Performance:</strong> Comparison between cameras</li>
                <li><strong>Product Distribution:</strong> Breakdown by product code</li>
                <li><strong>Hourly Statistics:</strong> Inspection volume by hour</li>
              </ul>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>View Historical Data:</strong> All users</p>
              <p><strong>Export Data:</strong> Admin and Manager only</p>
            </div>
          </div>
        );

      case 'settings':
        return (
          <div className="doc-content">
            <h1>⚙️ Settings</h1>
            <p className="doc-intro">
              Customize system preferences, loading animations, and configuration options.
            </p>

            <div className="doc-section">
              <h2>🎨 Loading Animation Templates</h2>
              <p>Customize loading animations for different sections:</p>
              <h3>Recipe Load Animation</h3>
              <p>Choose the animation displayed when loading a recipe:</p>
              <ul>
                <li><strong>OCR Scanner:</strong> Text recognition scanning effect</li>
                <li><strong>Camera Vision:</strong> Grid-based vision system</li>
                <li><strong>Barcode Scanner:</strong> Laser barcode detection</li>
                <li><strong>Neural Network:</strong> AI processing visualization</li>
                <li><strong>QR Detector:</strong> QR code detection system</li>
                <li><strong>Industrial Factory:</strong> Gear system with LED indicators</li>
              </ul>

              <h3>Loading Overlay Mode</h3>
              <p>Control where the loading animation appears:</p>
              <ul>
                <li><strong>Full Screen:</strong> Cover entire window (including sidebar and header)</li>
                <li><strong>Dashboard Main:</strong> Cover content area only (sidebar remains visible)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📄 Page Loading Templates</h2>
              <p>Customize loading animations for each page navigation:</p>
              <ul>
                <li>Dashboard Loading</li>
                <li>User Management Loading</li>
                <li>Recipes Loading</li>
                <li>Cameras Loading</li>
                <li>Historical Loading</li>
                <li>Settings Loading</li>
                <li>Logs Loading</li>
              </ul>
              <p>Choose from various animation styles including Camera Vision, Spinner, Pulse, Radar, Grid, Circuit, etc.</p>
            </div>

            <div className="doc-section">
              <h2>🎭 Theme Settings</h2>
              <ul>
                <li><strong>Dark Mode:</strong> Toggle dark/light theme (header toggle)</li>
                <li><strong>Loading Background:</strong> Select background pattern for loading screens</li>
                <li><strong>Background Opacity:</strong> Adjust transparency (0.0 - 1.0)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🎥 Camera Settings</h2>
              <p>Configure default camera parameters:</p>
              <ul>
                <li><strong>Camera 1/2/3 Enabled:</strong> Enable/disable each camera</li>
                <li><strong>FPS:</strong> Default frame rate (10-60 FPS)</li>
                <li><strong>Resolution:</strong> Default resolution (e.g., 1920×1080)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🔧 Processing Settings</h2>
              <ul>
                <li><strong>Detection Threshold:</strong> Confidence level for OCR (0.0 - 1.0)</li>
                <li><strong>Max Processing Time:</strong> Timeout in seconds</li>
                <li><strong>Auto Reject:</strong> Automatically reject low-confidence results</li>
                <li><strong>Save Failed Images:</strong> Store images of failed inspections</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🔔 Notification Settings</h2>
              <ul>
                <li><strong>Email Notifications:</strong> Send alerts via email</li>
                <li><strong>SMS Notifications:</strong> Send alerts via SMS</li>
                <li><strong>Alert Threshold:</strong> Number of failures before alert</li>
                <li><strong>Daily Report:</strong> Receive daily summary email</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>💾 Saving Settings</h2>
              <p>Click <strong>"Save Settings"</strong> at the bottom to apply changes. Most settings take effect immediately, but some may require a page refresh or system restart.</p>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>View Settings:</strong> All users</p>
              <p><strong>Modify Settings:</strong> Admin only</p>
              <p><strong>Note:</strong> Regular users can view settings but cannot save changes.</p>
            </div>
          </div>
        );

      case 'logs':
        return (
          <div className="doc-content">
            <h1>📝 Activity Logs</h1>
            <p className="doc-intro">
              View comprehensive audit trail of all system activities, user actions, and system events.
            </p>

            <div className="doc-section">
              <h2>📋 Log Types</h2>
              <div className="log-types-grid">
                <div className="log-type">
                  <span className="badge blue">INFO</span>
                  <p>General information and successful operations</p>
                </div>
                <div className="log-type">
                  <span className="badge yellow">WARNING</span>
                  <p>Potential issues that don't prevent operation</p>
                </div>
                <div className="log-type">
                  <span className="badge red">ERROR</span>
                  <p>Errors and failures that need attention</p>
                </div>
                <div className="log-type">
                  <span className="badge green">SUCCESS</span>
                  <p>Successfully completed operations</p>
                </div>
              </div>
            </div>

            <div className="doc-section">
              <h2>🔍 Filtering Logs</h2>
              <p>Use filters to find specific log entries:</p>
              <ul>
                <li><strong>Date Range:</strong> Filter by time period</li>
                <li><strong>User:</strong> Show actions by specific user</li>
                <li><strong>Action Type:</strong> Filter by operation (Create, Update, Delete, Load, etc.)</li>
                <li><strong>Log Level:</strong> Filter by severity (Info, Warning, Error, Success)</li>
                <li><strong>Search:</strong> Free-text search in log messages</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📊 Log Information</h2>
              <p>Each log entry contains:</p>
              <ul>
                <li><strong>Timestamp:</strong> Exact date and time of the event</li>
                <li><strong>User:</strong> Who performed the action (or "System" for automated actions)</li>
                <li><strong>Action:</strong> What operation was performed</li>
                <li><strong>Target:</strong> Which entity was affected (user, recipe, camera, etc.)</li>
                <li><strong>Details:</strong> Additional information about the action</li>
                <li><strong>Result:</strong> Success or failure status</li>
                <li><strong>IP Address:</strong> Source IP of the action (if applicable)</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>📥 Exporting Logs</h2>
              <ol>
                <li>Apply filters to select the logs you want to export</li>
                <li>Click <strong>"Export Logs"</strong></li>
                <li>Choose format:
                  <ul>
                    <li><strong>Excel (.xlsx):</strong> Formatted spreadsheet</li>
                    <li><strong>CSV (.csv):</strong> Raw comma-separated values</li>
                    <li><strong>JSON (.json):</strong> Structured data format</li>
                  </ul>
                </li>
                <li>Download the file</li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>🔎 Common Log Actions</h2>
              <table className="log-actions-table">
                <thead>
                  <tr>
                    <th>Action</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>LOGIN</strong></td>
                    <td>User logged into the system</td>
                  </tr>
                  <tr>
                    <td><strong>LOGOUT</strong></td>
                    <td>User logged out</td>
                  </tr>
                  <tr>
                    <td><strong>CREATE_USER</strong></td>
                    <td>New user account created</td>
                  </tr>
                  <tr>
                    <td><strong>UPDATE_USER</strong></td>
                    <td>User information modified</td>
                  </tr>
                  <tr>
                    <td><strong>DELETE_USER</strong></td>
                    <td>User account deleted</td>
                  </tr>
                  <tr>
                    <td><strong>CREATE_RECIPE</strong></td>
                    <td>New recipe created</td>
                  </tr>
                  <tr>
                    <td><strong>LOAD_RECIPE</strong></td>
                    <td>Recipe loaded for production</td>
                  </tr>
                  <tr>
                    <td><strong>UPDATE_SETTINGS</strong></td>
                    <td>System settings changed</td>
                  </tr>
                  <tr>
                    <td><strong>CAMERA_CONNECT</strong></td>
                    <td>Camera connected to system</td>
                  </tr>
                  <tr>
                    <td><strong>INSPECTION_FAIL</strong></td>
                    <td>Product failed inspection</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="doc-section">
              <h2>⚠️ Important Notes</h2>
              <ul>
                <li>Logs are retained for 90 days by default</li>
                <li>Critical actions (user deletion, recipe loading) are permanently logged</li>
                <li>System automatically logs all user actions for audit compliance</li>
                <li>Logs cannot be modified or deleted by users</li>
              </ul>
            </div>

            <div className="doc-section permission-required">
              <h2>🔒 Permissions Required</h2>
              <p><strong>View Logs:</strong> Admin only</p>
              <p>Supervisors and Operators cannot access the Logs page.</p>
            </div>
          </div>
        );

      case 'permissions':
        return (
          <div className="doc-content">
            <h1>🔐 Permissions & Access Control</h1>
            <p className="doc-intro">
              Understanding user roles, permissions, and access levels in the system.
            </p>

            <div className="doc-section">
              <h2>👤 User Roles</h2>
              <p>The system has three primary user roles:</p>

              <div className="role-card admin">
                <div className="role-header">
                  <h3>🔴 Administrator</h3>
                  <span className="badge red">ADMIN</span>
                </div>
                <p className="role-description">
                  Full system access with all permissions. Can manage users, configure system settings, access all features, and change passwords for Operators and Supervisors.
                </p>
                <h4>Permissions:</h4>
                <ul className="permission-list">
                  <li>✅ View Dashboard</li>
                  <li>✅ Manage Users (Create, Edit, Delete)</li>
                  <li>✅ Change Passwords (for Operators and Supervisors)</li>
                  <li>✅ Manage Recipes (Create, Update, Delete, Load)</li>
                  <li>✅ Edit and Delete Templates</li>
                  <li>✅ Manage Cameras (Create, Edit, Delete, Connect)</li>
                  <li>✅ View Historical Data</li>
                  <li>✅ Configure System Settings</li>
                  <li>✅ View Logs</li>
                  <li>✅ View Documentation</li>
                </ul>
              </div>

              <div className="role-card manager">
                <div className="role-header">
                  <h3>🟡 Supervisor</h3>
                  <span className="badge yellow">SUPERVISOR</span>
                </div>
                <p className="role-description">
                  Supervisory access for production management. Can create and update recipes, but cannot delete them or access system settings.
                </p>
                <h4>Permissions:</h4>
                <ul className="permission-list">
                  <li>✅ View Dashboard</li>
                  <li>✅ Create Recipes</li>
                  <li>✅ Update Recipes (Edit existing recipes)</li>
                  <li>✅ Load Recipes</li>
                  <li>✅ Edit Templates</li>
                  <li>✅ View Historical Data</li>
                  <li>✅ View Documentation</li>
                  <li>❌ Delete Recipes</li>
                  <li>❌ Delete Templates</li>
                  <li>❌ Access User Management</li>
                  <li>❌ Access Camera Management</li>
                  <li>❌ Access Settings</li>
                  <li>❌ Access Logs</li>
                </ul>
              </div>

              <div className="role-card operator">
                <div className="role-header">
                  <h3>🟢 Operator</h3>
                  <span className="badge green">OPERATOR</span>
                </div>
                <p className="role-description">
                  Basic production access. Can view data, load recipes, and edit recipe templates (Template tab only), but cannot create, update, or delete recipes.
                </p>
                <h4>Permissions:</h4>
                <ul className="permission-list">
                  <li>✅ View Dashboard</li>
                  <li>✅ View Recipes</li>
                  <li>✅ Load Recipes</li>
                  <li>✅ Edit Recipe Templates (Template tab only - can draw on templates)</li>
                  <li>✅ View Historical Data</li>
                  <li>✅ View Documentation</li>
                  <li>❌ Create Recipes</li>
                  <li>❌ Update Recipe Basic Info</li>
                  <li>❌ Update Recipe Camera Settings</li>
                  <li>❌ Update Recipe Model Settings</li>
                  <li>❌ Delete Recipes</li>
                  <li>❌ Delete Templates</li>
                  <li>❌ Access User Management</li>
                  <li>❌ Access Camera Management</li>
                  <li>❌ Access Settings</li>
                  <li>❌ Access Logs</li>
                </ul>
              </div>
            </div>

            <div className="doc-section">
              <h2>🔑 Granular Permissions</h2>
              <p>The system uses the following specific permissions:</p>
              <table className="permissions-table">
                <thead>
                  <tr>
                    <th>Permission</th>
                    <th>Description</th>
                    <th>Roles with Access</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>view_dashboard</code></td>
                    <td>Access Dashboard page</td>
                    <td>Admin, Supervisor, Operator</td>
                  </tr>
                  <tr>
                    <td><code>view_historical</code></td>
                    <td>View historical data and reports</td>
                    <td>Admin, Supervisor, Operator</td>
                  </tr>
                  <tr>
                    <td><code>view_user_management</code></td>
                    <td>Access User Management page</td>
                    <td>Admin only</td>
                  </tr>
                  <tr>
                    <td><code>view_camera_management</code></td>
                    <td>Access Camera Management page</td>
                    <td>Admin only</td>
                  </tr>
                  <tr>
                    <td><code>view_settings</code></td>
                    <td>Access Settings page</td>
                    <td>Admin only</td>
                  </tr>
                  <tr>
                    <td><code>view_logs</code></td>
                    <td>Access Activity Logs page</td>
                    <td>Admin only</td>
                  </tr>
                  <tr>
                    <td><code>create_receipt</code></td>
                    <td>Create new recipes</td>
                    <td>Admin, Supervisor</td>
                  </tr>
                  <tr>
                    <td><code>update_receipt</code></td>
                    <td>Update existing recipes (Basic Info, Camera, Model tabs)</td>
                    <td>Admin, Supervisor</td>
                  </tr>
                  <tr>
                    <td><code>delete_receipt</code></td>
                    <td>Delete recipes</td>
                    <td>Admin only</td>
                  </tr>
                  <tr>
                    <td><code>load_receipt</code></td>
                    <td>Load recipes for production</td>
                    <td>Admin, Supervisor, Operator</td>
                  </tr>
                  <tr>
                    <td><code>edit_template</code></td>
                    <td>Edit recipe templates (Template tab - draw on templates)</td>
                    <td>Admin, Supervisor, Operator</td>
                  </tr>
                  <tr>
                    <td><code>delete_template</code></td>
                    <td>Delete templates from recipes</td>
                    <td>Admin, Supervisor</td>
                  </tr>
                  <tr>
                    <td><code>change_passwords</code></td>
                    <td>Change passwords for Operators and Supervisors</td>
                    <td>Admin only</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="doc-section">
              <h2>🚫 Access Denied Scenarios</h2>
              <p>What happens when you try to access a restricted feature:</p>
              <ul>
                <li><strong>Page Access:</strong> If you navigate to a page you don't have permission for, you'll see an "Access Denied" message</li>
                <li><strong>Menu Items:</strong> Restricted pages are hidden from the sidebar menu</li>
                <li><strong>Action Buttons:</strong> Buttons for actions you can't perform are disabled or hidden</li>
                <li><strong>API Calls:</strong> Server validates permissions; unauthorized requests return 403 Forbidden</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>🔒 Security Best Practices</h2>
              <ul>
                <li>✅ Use strong passwords (minimum 6 characters, mix of letters and numbers)</li>
                <li>✅ Don't share your login credentials</li>
                <li>✅ Log out when finished using the system</li>
                <li>✅ Report suspicious activity to administrators</li>
                <li>✅ Keep your account information up to date</li>
                <li>❌ Don't leave your workstation unlocked</li>
                <li>❌ Don't use public/shared computers for accessing the system</li>
                <li>❌ Don't write down passwords</li>
              </ul>
            </div>

            <div className="doc-section">
              <h2>👥 Requesting Permissions</h2>
              <p>If you need additional permissions:</p>
              <ol>
                <li>Identify which permission or role you need</li>
                <li>Contact your system administrator or manager</li>
                <li>Explain why you need the additional access</li>
                <li>Wait for approval and account update</li>
                <li>Log out and log back in to activate new permissions</li>
              </ol>
            </div>

            <div className="doc-section">
              <h2>📊 Permission Matrix</h2>
              <div className="matrix-scroll">
                <table className="permission-matrix">
                  <thead>
                    <tr>
                      <th>Feature / Action</th>
                      <th>Admin</th>
                      <th>Supervisor</th>
                      <th>Operator</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>View Dashboard</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                    </tr>
                    <tr>
                      <td>View Documentation</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                    </tr>
                    <tr>
                      <td>View Historical Data</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                    </tr>
                    <tr>
                      <td>Access User Management</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Access Camera Management</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Access Settings</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Access Logs</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Create Recipes</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Update Recipes (Basic Info, Camera, Model)</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Load Recipes</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                    </tr>
                    <tr>
                      <td>Edit Templates (Template tab only)</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                      <td className="granted">✅</td>
                    </tr>
                    <tr>
                      <td>Delete Recipes</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Delete Templates</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                    <tr>
                      <td>Change User Passwords</td>
                      <td className="granted">✅</td>
                      <td className="denied">❌</td>
                      <td className="denied">❌</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="note">
                <strong>Note:</strong> Operators can click "Edit" on recipes but can only interact with the Template tab.
                The Basic Info, Camera, and Model tabs are read-only for Operators.
              </p>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="documentation-page">
      <div className="doc-sidebar">
        <div className="doc-sidebar-header">
          <h2>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }}>
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M6.5 2H20V22H6.5A2.5 2.5 0 0 1 4 19.5V4.5A2.5 2.5 0 0 1 6.5 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Documentation
          </h2>
          <p>System Guide</p>
        </div>

        <nav className="doc-nav">
          <button
            className={`doc-nav-item ${activeSection === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveSection('overview')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M9 22V12H15V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Overview
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveSection('dashboard')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Dashboard
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'users' ? 'active' : ''}`}
            onClick={() => setActiveSection('users')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="9" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
              <path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M16 3.13C16.8604 3.35031 17.623 3.85071 18.1676 4.55232C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89318 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            User Management
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'recipes' ? 'active' : ''}`}
            onClick={() => setActiveSection('recipes')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="10,9 9,9 8,9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Recipes
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'cameras' ? 'active' : ''}`}
            onClick={() => setActiveSection('cameras')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M7 4v2M17 4v2" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Cameras
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'historical' ? 'active' : ''}`}
            onClick={() => setActiveSection('historical')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
              <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Historical Data
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveSection('settings')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M19.4 15A1.65 1.65 0 0 0 21 13.35A1.65 1.65 0 0 0 19.4 11.65L18.7 12.35A7.81 7.81 0 0 0 12 5.29V4A1 1 0 0 0 11 3A1 1 0 0 0 10 4V5.29A7.81 7.81 0 0 0 3.3 12.35L2.6 11.65A1.65 1.65 0 0 0 1 13.35A1.65 1.65 0 0 0 2.6 15L3.3 14.3A7.81 7.81 0 0 0 10 20.71V22A1 1 0 0 0 11 23A1 1 0 0 0 12 22V20.71A7.81 7.81 0 0 0 18.7 14.3L19.4 15Z" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Settings
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveSection('logs')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M9 12H15M9 16H15M17 21H7C5.89543 21 5 20.1046 5 19V5C5 3.89543 5.89543 3 7 3H12L19 10V19C19 20.1046 18.1046 21 17 21Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 3V10H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Activity Logs
          </button>
          <button
            className={`doc-nav-item ${activeSection === 'permissions' ? 'active' : ''}`}
            onClick={() => setActiveSection('permissions')}
          >
            <svg className="doc-nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="11" width="18" height="11" rx="2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M7 11V7C7 5.67392 7.52678 4.40215 8.46447 3.46447C9.40215 2.52678 10.6739 2 12 2C13.3261 2 14.5979 2.52678 15.5355 3.46447C16.4732 4.40215 17 5.67392 17 7V11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Permissions
          </button>
        </nav>
      </div>

      <div className="doc-main">
        {renderContent()}
      </div>
    </div>
  );
};

export default Documentation;
