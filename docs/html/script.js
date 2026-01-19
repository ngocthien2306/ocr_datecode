// Sample data for camera charts
const camera1Data = [45, 52, 48, 65, 58, 72, 68, 75, 82, 78, 85, 92];
const camera2Data = [38, 42, 45, 52, 48, 58, 62, 68, 65, 72, 75, 80];
const camera3Data = [32, 35, 38, 42, 45, 48, 52, 55, 58, 62, 65, 68];
const hours = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'];

// Draw simple line chart
function drawLineChart(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  
  // Clear canvas
  ctx.clearRect(0, 0, width, height);
  
  // Set up chart dimensions
  const padding = 40;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;
  
  // Find max and min values
  const maxValue = Math.max(...data);
  const minValue = Math.min(...data);
  const valueRange = maxValue - minValue || 1;
  
  // Draw grid lines
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  
  // Horizontal grid lines
  for (let i = 0; i <= 4; i++) {
    const y = padding + (chartHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }
  
  // Draw line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  
  data.forEach((value, index) => {
    const x = padding + (index / (data.length - 1)) * chartWidth;
    const y = padding + chartHeight - ((value - minValue) / valueRange) * chartHeight;
    
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  
  ctx.stroke();
  
  // Draw area under the line
  ctx.lineTo(padding + chartWidth, padding + chartHeight);
  ctx.lineTo(padding, padding + chartHeight);
  ctx.closePath();
  
  const gradient = ctx.createLinearGradient(0, padding, 0, padding + chartHeight);
  gradient.addColorStop(0, color + '40');
  gradient.addColorStop(1, color + '00');
  ctx.fillStyle = gradient;
  ctx.fill();
  
  // Draw points
  ctx.fillStyle = color;
  data.forEach((value, index) => {
    const x = padding + (index / (data.length - 1)) * chartWidth;
    const y = padding + chartHeight - ((value - minValue) / valueRange) * chartHeight;
    
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    
    // Add white border to points
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  
  // Draw axis labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'center';
  
  // X-axis labels (time)
  const labelStep = Math.ceil(hours.length / 6);
  hours.forEach((hour, index) => {
    if (index % labelStep === 0 || index === hours.length - 1) {
      const x = padding + (index / (data.length - 1)) * chartWidth;
      ctx.fillText(hour, x, height - 15);
    }
  });
  
  // Y-axis labels (values)
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const value = minValue + (valueRange / 4) * (4 - i);
    const y = padding + (chartHeight / 4) * i + 5;
    ctx.fillText(Math.round(value), padding - 10, y);
  }
}

// Navigation functionality
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item[data-section]');
  const sections = document.querySelectorAll('.content-section');
  
  navItems.forEach(item => {
    item.addEventListener('click', function(e) {
      e.preventDefault();
      
      const sectionId = this.getAttribute('data-section');
      
      // Remove active class from all nav items
      navItems.forEach(nav => nav.classList.remove('active'));
      
      // Add active class to clicked item
      this.classList.add('active');
      
      // Hide all sections
      sections.forEach(section => section.classList.remove('active'));
      
      // Show selected section
      const targetSection = document.getElementById(sectionId + '-section');
      if (targetSection) {
        targetSection.classList.add('active');
      }
    });
  });
}

// Dark mode toggle
function setupDarkMode() {
  const darkModeToggle = document.querySelector('.dark-mode-toggle input');
  
  if (darkModeToggle) {
    // Check if user has a preference saved
    const darkModePreference = localStorage.getItem('darkMode');
    if (darkModePreference === 'enabled') {
      document.body.classList.add('dark-mode');
      darkModeToggle.checked = true;
    }
    
    darkModeToggle.addEventListener('change', function() {
      if (this.checked) {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'enabled');
      } else {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'disabled');
      }
      
      // Redraw charts with updated colors
      setTimeout(() => {
        drawAllCharts();
      }, 100);
    });
  }
}

// Logout functionality
function setupLogout() {
  const logoutBtn = document.getElementById('logoutBtn');
  
  if (logoutBtn) {
    logoutBtn.addEventListener('click', function(e) {
      e.preventDefault();
      if (confirm('Are you sure you want to logout?')) {
        console.log('Logging out...');
        // Here you would typically redirect to login page or clear session
        alert('Logout functionality will be implemented');
      }
    });
  }
}

// Update time
function updateTime() {
  const timeElement = document.querySelector('.time');
  const dateElement = document.querySelector('.date');
  
  if (timeElement && dateElement) {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
    const dateString = now.toLocaleDateString('en-US', { 
      weekday: 'long', 
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    });
    
    timeElement.textContent = timeString;
    dateElement.textContent = dateString;
  }
}

// Draw all charts
function drawAllCharts() {
  const isDarkMode = document.body.classList.contains('dark-mode');
  const color1 = '#6366f1';
  const color2 = '#3b82f6';
  const color3 = '#8b5cf6';
  
  drawLineChart('camera1Chart', camera1Data, color1);
  drawLineChart('camera2Chart', camera2Data, color2);
  drawLineChart('camera3Chart', camera3Data, color3);
}

// Simulate real-time data updates
function simulateDataUpdates() {
  setInterval(() => {
    // Update total products today
    const totalProductsElement = document.getElementById('totalProductsToday');
    if (totalProductsElement) {
      const currentValue = parseInt(totalProductsElement.textContent.replace(',', ''));
      const newValue = currentValue + Math.floor(Math.random() * 5);
      totalProductsElement.textContent = newValue.toLocaleString();
    }
    
    // Update camera data
    camera1Data.shift();
    camera1Data.push(Math.floor(Math.random() * 40) + 60);
    
    camera2Data.shift();
    camera2Data.push(Math.floor(Math.random() * 35) + 55);
    
    camera3Data.shift();
    camera3Data.push(Math.floor(Math.random() * 30) + 45);
    
    // Redraw charts
    drawAllCharts();
    
    console.log('Data updated at:', new Date().toLocaleTimeString());
  }, 30000); // Update every 30 seconds
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Draw initial charts
  drawAllCharts();
  
  // Setup interactive features
  setupNavigation();
  setupDarkMode();
  setupLogout();
  
  // Update time initially and then every minute
  updateTime();
  setInterval(updateTime, 60000);
  
  // Start data simulation after 5 seconds
  setTimeout(simulateDataUpdates, 5000);
  
  console.log('Camera Management Dashboard initialized successfully!');
});

// Handle window resize to redraw charts
let resizeTimer;
window.addEventListener('resize', function() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(function() {
    drawAllCharts();
  }, 250);
});