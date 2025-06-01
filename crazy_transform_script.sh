
#!/bin/bash

echo "🚀 Starting Crazy Transformation Script..."

# 1) Remove any old copy and clone fresh
rm -rf Nr1Replit
git clone https://github.com/EliAlas11/Nr1Replit.git
cd Nr1Replit

# 2) Install Python requirements (if you have requirements.txt)
if [ -f requirements.txt ]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
fi

# 3) Install Node.js dependencies (if you have package.json)
if [ -f package.json ]; then
  npm install
fi

# 4) Install "crazy" global tools for formatting, minifying, docs
npm install -g prettier jsdoc terser csso || true
python3 -m pip install black || true

# 5) Format all Python files (if black is available)
if command -v black >/dev/null 2>&1; then
  black .
else
  echo "⚠️  black not found; skipping Python formatting"
fi

# 6) Format all JS/TS/JSON/MD files (if prettier is available)
if command -v prettier >/dev/null 2>&1; then
  prettier --write "**/*.{js,ts,json,md}" || true
else
  echo "⚠️  prettier not found; skipping JS/TS/JSON/MD formatting"
fi

# 7) Minify every .js (except in node_modules) into .min.js
find . -type f -name "*.js" -not -path "./node_modules/*" | while read -r file; do
  terser "$file" -o "${file%.js}.min.js" || true
done

# 8) Minify every .css (except in node_modules) into .min.css
find . -type f -name "*.css" -not -path "./node_modules/*" | while read -r file; do
  csso "$file" -o "${file%.css}.min.css" || true
done

# 9) If you have JSDoc configured, generate docs into ./out_docs
if [ -f package.json ]; then
  mkdir -p out_docs
  jsdoc -c jsdoc.json -d out_docs . || echo "⚠️  JSDoc failed or no jsdoc.json present"
fi

# 10) Return to parent folder and zip up the entire "crazy" transformed folder
cd ..
zip -r Nr1Replit_crazy_transformed.zip Nr1Replit

echo
echo "✅ Transformation complete!"
echo "▶️  Download your ZIP under the Files sidebar as: Nr1Replit_crazy_transformed.zip"
