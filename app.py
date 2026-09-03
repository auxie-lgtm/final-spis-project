import os
import tempfile
from flask import Flask, render_template, request
from ai import PromptManager
from karaoke_classifier import KaraokeClassifier

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'singer_grade_model.keras')
prompt_manager = None


def ensure_dataset_model():
    if os.path.exists(MODEL_PATH):
        print(f"Using cached model: {MODEL_PATH}")
        return
    print('Training karaoke model before app startup...')
    classifier = KaraokeClassifier(force_retrain=False)
    classifier.find_files()
    classifier.train()


def get_prompt_manager():
    global prompt_manager
    if prompt_manager is None:
        prompt_manager = PromptManager()
    return prompt_manager


@app.route('/')
def render_home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def render_result():
    try:
        audio_file = request.files.get('audio_file')
        message = request.form.get('message') or request.args.get('message') or ''

        if audio_file is None or not audio_file.filename:
            raise ValueError('No audio file was uploaded.')

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name

        prompt_manager = get_prompt_manager()
        response = prompt_manager.evaluate_song(temp_path, message)
        grade = prompt_manager.get_overall_grade() or 'F'

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return render_template('result.html', response=response, grade=grade)
    except Exception as e:
        return render_template('result.html', response=str(e), grade='F')

if __name__ == '__main__':
    ensure_dataset_model()
    app.run(host='0.0.0.0', debug=False)
