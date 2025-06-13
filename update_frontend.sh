#!/bin/bash
# Quick update script for frontend files

SOURCE_DIR="${1:-./www}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Source directory '$SOURCE_DIR' not found!"
    exit 1
fi

echo "🔄 Updating frontend files..."
sudo cp "$SOURCE_DIR/index.html" "/var/www/musequill-dev/" 2>/dev/null && echo "✅ index.html updated"
sudo cp "$SOURCE_DIR/style.css" "/var/www/musequill-dev/" 2>/dev/null && echo "✅ style.css updated"
sudo cp "$SOURCE_DIR/script.js" "/var/www/musequill-dev/" 2>/dev/null && echo "✅ script.js updated"

echo "🎉 Frontend updated successfully!"
echo "🌐 View at: http://localhost:8088"
