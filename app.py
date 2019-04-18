from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

from metrics import score
import os

from flask import jsonify

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

def test_input_validity(y_pred, y_true, y_score, k, method):
    has_only_lists = any([type(y_pred) != list, type(y_true) != list, type(y_score) != list])
    if has_only_lists:
        return "y_pred, y_true and  y_score should be arrays of integers"

    if not all([type(i) == int for i in y_score]):
        return "y_score can only contains integers"

    if type(k) != int:
        return "k should be integer"

    if k > len(y_pred) or k > len(y_true):
        return "k cannot be longer than y_pred"
    if method not in ["precision", "recall", "dcg", "mrr", "all"]:
        return 'method should be one of ["precision", "recall", "dcg", "mrr", "all"]'
    return False


def add_scoring(app):
    @app.route('/api/score', methods=['POST'])
    @cross_origin()

    def scoreit():
        
        data = request.json
        y_true = data.get('y_true')
        y_score = data.get('y_score')
        y_pred = data.get('y_pred')
        k = data.get('k')
        method = data.get('method')
        message = test_input_validity(y_pred, y_true, y_score, k, method)
        if message:
            raise InvalidUsage(message, status_code=422)

        results = score(y_pred, y_true, y_score, k, method)
        return jsonify(results)
    
    @app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def create_app():
  app = Flask(__name__)
  app.config['JSON_AS_ASCII'] = False
  app.logger.info("Flask app started ")

  @app.route('/')
  def hello():
    return 'scoring api'

  add_scoring(app)

  return app

if __name__ == "__main__":
    app.run(host='0.0.0.0')