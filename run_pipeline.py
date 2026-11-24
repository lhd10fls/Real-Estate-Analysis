"""Run preprocessing, training, evaluation and tests with the active interpreter."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = [
    ['data_preprocessed/rebuild_dataset.py'],
    ['-m','unittest','discover','-s','data_preprocessed','-p','test_rebuild.py','-v'],
    ['train.py'],
    ['-m','unittest','discover','-s','tests','-v'],
]

if __name__ == '__main__':
    for step in STEPS:
        print('Running:', ' '.join(step), flush=True)
        subprocess.run([sys.executable,'-X','utf8',*step],cwd=ROOT,check=True)
    print('Complete. Run "streamlit run streamlit_app.py" to open the application.')
