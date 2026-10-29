#!/bin/zsh
cd "$HOME/Downloads/MediGuard_AI_Ultimate" || { echo 'Project folder not found'; read '?Press Enter to close'; exit 1; }
if [[ ! -f .venv/bin/activate ]]; then echo 'Missing virtual environment'; read '?Press Enter to close'; exit 1; fi
source .venv/bin/activate
open 'http://localhost:8501'
python -m streamlit run app.py
