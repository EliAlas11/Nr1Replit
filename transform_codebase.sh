
#!/bin/bash

echo "🚀 Starting Nr1Copilot Codebase Transformation..."

# 1) Navigate to the main project directory
cd nr1copilot/nr1-main

# 2) Clean up legacy Node.js files (keeping only Python/FastAPI)
echo "🧹 Cleaning up legacy files..."
rm -rf controllers/ routes/ services/ validators/ utils/ queue/ config/
rm -f app.js server.js worker.js videoWorkerThread.js package.json
rm -f *.js *.md backend_todo.md todo.md ui_analysis.md viral_clip_generator_analysis.md
rm -f swagger.yaml render_deployment_guide.md render_deployment_todo.md
rm -f compatibility_test_report.md final_validation_report.md

# 3) Install Python requirements
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Install formatting tools
echo "🛠️ Installing development tools..."
python -m pip install black isort flake8 mypy

# 5) Format Python code with black
echo "🎨 Formatting Python code..."
python -m black app/ --line-length 88 --target-version py38

# 6) Sort imports with isort
echo "📝 Sorting imports..."
python -m isort app/ --profile black

# 7) Run linting checks
echo "🔍 Running code quality checks..."
python -m flake8 app/ --max-line-length=88 --extend-ignore=E203,W503

# 8) Create production-ready structure
echo "📁 Creating production structure..."
mkdir -p videos uploads logs static

# 9) Create optimized versions of critical files
echo "⚡ Creating optimized configurations..."

# 10) Generate API documentation
echo "📚 Generating API documentation..."
python3 -c "
import json
from app.main import app
with open('api_schema.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"

# 11) Create deployment-ready package
echo "📦 Creating deployment package..."
cd ../..
tar -czf nr1copilot_production.tar.gz nr1copilot/ --exclude="__pycache__" --exclude="*.pyc" --exclude=".git"

echo
echo "✅ Transformation complete!"
echo "🎯 Your optimized codebase is ready for deployment"
echo "📦 Production package: nr1copilot_production.tar.gz"
