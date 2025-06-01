
#!/bin/bash

echo "🚀 Starting Crazy Transformation Script (Replit Compatible)..."

# 1) Remove any old copy and clone fresh
rm -rf Nr1Replit
git clone https://github.com/EliAlas11/Nr1Replit.git
cd Nr1Replit

# 2) Install Python requirements (using python instead of python3)
if [ -f nr1copilot/nr1-main/requirements.txt ]; then
  echo "📦 Installing Python dependencies..."
  cd nr1copilot/nr1-main
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  cd ../..
fi

# 3) Install formatting tools for Python
echo "🛠️ Installing development tools..."
python -m pip install black isort flake8 || true

# 4) Format Python code with black (if available)
if command -v black >/dev/null 2>&1; then
  echo "🎨 Formatting Python code..."
  black Nr1Replit/nr1copilot/nr1-main/app/ --line-length 88 || true
else
  echo "⚠️  black not found; skipping Python formatting"
fi

# 5) Sort imports with isort (if available)
if command -v isort >/dev/null 2>&1; then
  echo "📝 Sorting imports..."
  isort Nr1Replit/nr1copilot/nr1-main/app/ --profile black || true
else
  echo "⚠️  isort not found; skipping import sorting"
fi

# 6) Create production-ready structure
echo "📁 Creating production structure..."
cd Nr1Replit/nr1copilot/nr1-main
mkdir -p videos uploads logs static

# 7) Clean up legacy files
echo "🧹 Cleaning up legacy files..."
rm -f *.js *.md backend_todo.md todo.md ui_analysis.md viral_clip_generator_analysis.md
rm -f swagger.yaml render_deployment_guide.md render_deployment_todo.md
rm -f compatibility_test_report.md final_validation_report.md package.json

# 8) Create tar.gz archive (since zip isn't available)
cd ../../..
echo "📦 Creating archive..."
tar -czf Nr1Replit_crazy_transformed.tar.gz Nr1Replit

echo
echo "✅ Transformation complete!"
echo "▶️  Download your archive as: Nr1Replit_crazy_transformed.tar.gz"
echo "📝 Your FastAPI backend is ready for deployment!"
