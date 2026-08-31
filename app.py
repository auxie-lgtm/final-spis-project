from flask import Flask, render_template
from karaoke_classifier import *
app = Flask(__name__)

@app.before_first_request
def initialize_model():
    k = KaraokeClassifier()
    k.find_files()
    k.train()

@app.route('/')
def render_home():
    return render_template('index.html')

@app.route('/eval')
def render_evaluation():
    return render_template('eval.html')



if __name__ == "__main__":
   app.run(host='0.0.0.0')