from flask import Flask, request, Response
from flask_restful import Resource,Api 
import pickle 
import pandas as pd 
from flask_cors import CORS
import numpy as np 
import cv2 
import jsonpickle
from prediction import pred
app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})



class model(Resource):
    def __init__(self):
        pass

   
    def post(self):

        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #get result from method pred 
        result = pred(img)
        # build a response dict to send back to client
        response = {'class':result}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")


api.add_resource(model,'/')

if __name__ == "__main__":
    app.run(debug=True)