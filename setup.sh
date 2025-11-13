#!/usr/bin/env bash
set -e

# -------------------------
# CONFIG
# -------------------------
ENV_NAME="hilabs_env"
PYTHON_BIN="python3"   # change if needed: python3.10
REQ_FILE="requirements.txt"

# -------------------------
# Banner
# -------------------------
echo "──────────────────────────────────────────────"
echo "   HiLabs Local Environment Setup"
echo "──────────────────────────────────────────────"

# -------------------------
# Python Check
# -------------------------
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "❌ ERROR: $PYTHON_BIN not found. Install Python first."
    exit 1
fi

# -------------------------
# Create virtual environment
# -------------------------
echo "✅ Creating virtual environment: $ENV_NAME"
$PYTHON_BIN -m venv $ENV_NAME

# -------------------------
# Activate environment
# -------------------------
echo "✅ Activating environment"
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# -------------------------
# Upgrade pip
# -------------------------
echo "✅ Upgrading pip"
pip install --upgrade pip

# -------------------------
# Install dependencies
# -------------------------
echo "✅ Installing dependencies"

pip install \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    streamlit \
    matplotlib \
    seaborn \
    joblib \
    python-dotenv

# -------------------------
# Freeze requirements
# -------------------------
echo "✅ Creating requirements.txt"
pip freeze > "$REQ_FILE"

# -------------------------
# Complete
# -------------------------
echo ""
echo "🎉 Setup complete!"
echo "──────────────────────────────────────────────"
echo "To activate this environment:"
echo ""
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "To run Streamlit:"
echo ""
echo "   streamlit run streamlit_app.py"
echo ""
echo "To install from requirements next time:"
echo ""
echo "   pip install -r $REQ_FILE"
echo "──────────────────────────────────────────────"
