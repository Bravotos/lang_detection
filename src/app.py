from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import socket
import pickle

from pathlib import Path
BASE_DIR=Path(__file__).resolve(strict=True).parent
#print(BASE_DIR, "YIPe")

app=Flask(__name__)
with open(rf'{BASE_DIR}/artifacts/trained_pipeline_0.1.0.pkl', 'rb') as f:
    model=pickle.load(f)
with open(rf'{BASE_DIR}/artifacts/labeler.pkl', 'rb') as f:
    labeler=pickle.load(f)

def pipeline_predict(model,x):
    a=model.predict(x)
    a=labeler.inverse_transform(a)
    return a

@app.route("/")
def index():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return render_template('index.html', hostname=host_name, ip=host_ip)
    except:
        return render_template('error.html')


@app.route('/submit', methods=['POST', 'GET'])
def predict():
    x = request.form['text']
    return render_template('index.html', pred=pipeline_predict(model,[x]))




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)