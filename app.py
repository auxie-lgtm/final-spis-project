from flask import Flask, render_template, jsonify, request
from ai import PromptManager


app = Flask(__name__)
manager = PromptManager()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/run-code', methods=['POST'])
def run_code():
    message = request.form.get('description', '')
    audio_path = request.form.get('audio_directory', '')

    if not message or not audio_path:
        return jsonify({"error": "Please provide both a description and an audio path."}), 400

    try:
        result = manager.prompt(message, audio_path)
        return jsonify({"output": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
