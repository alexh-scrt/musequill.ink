#!/bin/bash
# Streamlined MuseQuill Deployment Script
# Sets up Nginx configuration and copies existing files

set -e  # Exit on any error

echo "🚀 MuseQuill Deployment Setup"
echo "============================="

# Configuration
PROJECT_DIR="/var/www/musequill-dev"
NGINX_SITE="musequill-dev"
FRONTEND_PORT=8088
API_PORT=8055
SOURCE_DIR="${1:-./www}"  # Default to ./www directory, or use first argument

# Function to print colored output
print_status() {
    echo -e "\033[1;32m✅ $1\033[0m"
}

print_info() {
    echo -e "\033[1;34mℹ️  $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m❌ $1\033[0m"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory '$SOURCE_DIR' not found!"
    echo "Usage: $0 [source_directory]"
    echo "Example: $0 ./www"
    echo "Expected files: index.html, style.css, script.js"
    exit 1
fi

# Check if required files exist
required_files=("index.html" "style.css" "script.js")
for file in "${required_files[@]}"; do
    if [ ! -f "$SOURCE_DIR/$file" ]; then
        print_error "Required file '$SOURCE_DIR/$file' not found!"
        exit 1
    fi
done

print_info "Source files found in '$SOURCE_DIR'"

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    print_info "Installing Nginx..."
    sudo apt update
    sudo apt install -y nginx
    sudo systemctl enable nginx
    sudo systemctl start nginx
fi

print_status "Nginx is installed and running"

# Create project directory
print_info "Creating project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR

# Copy frontend files
print_info "Copying frontend files..."
sudo cp "$SOURCE_DIR/index.html" "$PROJECT_DIR/"
sudo cp "$SOURCE_DIR/style.css" "$PROJECT_DIR/"
sudo cp "$SOURCE_DIR/script.js" "$PROJECT_DIR/"

print_status "Frontend files copied"

# Create Nginx site configuration
print_info "Creating Nginx configuration..."

sudo tee /etc/nginx/sites-available/$NGINX_SITE > /dev/null << EOF
# MuseQuill Frontend Configuration
# Serves frontend on port $FRONTEND_PORT and proxies API calls to 127.0.0.1:$API_PORT

server {
    ssl_certificate /etc/letsencrypt/live/musequill.ink/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/musequill.ink/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot
    listen $FRONTEND_PORT;
    server_name localhost;
    
    # Root directory for frontend files
    root $PROJECT_DIR;
    index index.html;
    
    # Frontend static files
    location / {
        try_files \$uri \$uri/ /index.html;
        
        # Cache static assets
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
    }
    
    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://127.0.0.1:$API_PORT/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Handle preflight requests for CORS
        if (\$request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types
        application/atom+xml
        application/javascript
        application/json
        application/rss+xml
        application/vnd.ms-fontobject
        application/x-font-ttf
        application/x-web-app-manifest+json
        application/xhtml+xml
        application/xml
        font/opentype
        image/svg+xml
        image/x-icon
        text/css
        text/plain
        text/x-component;
    
    # Error pages
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    
    # Logging
    access_log /var/log/nginx/musequill_access.log;
    error_log /var/log/nginx/musequill_error.log;
}
EOF

print_status "Nginx configuration created"

# Enable the site
print_info "Enabling MuseQuill site..."
sudo ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/

# Test Nginx configuration
print_info "Testing Nginx configuration..."
if sudo nginx -t; then
    print_status "Nginx configuration is valid"
    
    # Reload Nginx
    print_info "Reloading Nginx..."
    sudo systemctl reload nginx
    print_status "Nginx reloaded successfully"
else
    print_error "Nginx configuration test failed"
    exit 1
fi

# Set proper permissions
print_info "Setting file permissions..."
sudo chown -R www-data:www-data $PROJECT_DIR
sudo chmod -R 755 $PROJECT_DIR

# Create a simple 404 page
sudo tee $PROJECT_DIR/404.html > /dev/null << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Page Not Found - MuseQuill</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; }
        .container { max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖋️ MuseQuill.ink</h1>
        <h2>Page Not Found</h2>
        <p>The page you're looking for doesn't exist.</p>
        <a href="/">← Back to Book Planner</a>
    </div>
</body>
</html>
EOF

# Create update script for easy redeployment
cat > update_frontend.sh << EOF
#!/bin/bash
# Quick update script for frontend files

SOURCE_DIR="\${1:-./www}"

if [ ! -d "\$SOURCE_DIR" ]; then
    echo "❌ Source directory '\$SOURCE_DIR' not found!"
    exit 1
fi

echo "🔄 Updating frontend files..."
sudo cp "\$SOURCE_DIR/index.html" "$PROJECT_DIR/" 2>/dev/null && echo "✅ index.html updated"
sudo cp "\$SOURCE_DIR/style.css" "$PROJECT_DIR/" 2>/dev/null && echo "✅ style.css updated"
sudo cp "\$SOURCE_DIR/script.js" "$PROJECT_DIR/" 2>/dev/null && echo "✅ script.js updated"

echo "🎉 Frontend updated successfully!"
echo "🌐 View at: http://localhost:$FRONTEND_PORT"
EOF

chmod +x update_frontend.sh

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "📊 Configuration Summary:"
echo "  Frontend URL: http://localhost:$FRONTEND_PORT"
echo "  Backend API:  http://127.0.0.1:$API_PORT"
echo "  Files location: $PROJECT_DIR"
echo "  Nginx config: /etc/nginx/sites-available/$NGINX_SITE"
echo ""
echo "🔧 Next Steps:"
echo ""
echo "1. Start your API server:"
echo "   python api.py --host 127.0.0.1 --port $API_PORT"
echo ""
echo "2. Test the deployment:"
echo "   curl http://localhost:$FRONTEND_PORT"
echo "   curl http://localhost:$FRONTEND_PORT/api/health"
echo ""
echo "3. Update frontend files anytime:"
echo "   ./update_frontend.sh [source_directory]"
echo ""
echo "📋 Useful Commands:"
echo "  sudo systemctl status nginx     # Check Nginx status"
echo "  sudo nginx -t                   # Test configuration"
echo "  sudo systemctl reload nginx    # Reload configuration"
echo "  sudo tail -f /var/log/nginx/musequill_error.log  # View error logs"
echo "  sudo tail -f /var/log/nginx/musequill_access.log # View access logs"
echo ""
echo "🌐 Your MuseQuill frontend is now live at: http://localhost:$FRONTEND_PORT"

# Quick connectivity test
if command -v curl &> /dev/null; then
    echo ""
    print_info "Testing frontend connectivity..."
    if curl -s http://localhost:$FRONTEND_PORT/health > /dev/null; then
        print_status "Frontend is responding!"
    else
        print_warning "Frontend not responding yet. Make sure Nginx is running."
    fi
fi