from flask import Flask, request, Response, abort, render_template

import matplotlib.pyplot as plt


class ModelValidator:

    def __init__(self):
        """
        Model Validator app
        """
        self.app = Flask(__name__)

        self.__metrics = None

        @self.app.route('/', methods=['POST', 'GET'])
        def root():
            return render_template('home/index.html')

    def run(self):
        self.app.run(port=5001)

if __name__ == '__main__':

    model_validator = ModelValidator()
    model_validator.run()