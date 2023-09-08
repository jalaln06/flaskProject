from flask import Flask, request
import modules
from modules.model import init_model
from modules.pdfreader import read_pdf
from modules.process import data_process
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/question/<question>')
@cross_origin()
def question(question):  # put application's code here
    data = data_process(question)
    return data


@app.route('/analyze')
@cross_origin()
def analyze():  # put application's code here
    data = modules.model.run_model_on_text( modules.pdfreader.read_pdf_from_url(request.args.get('article')))
    print(data)
    return {'summary': data}


if __name__ == '__main__':
    app.run()
    init_model()
