
from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
import uuid


class FLServer(object):
    MIN_NUM_WORKERS = 1
    MAX_NUM_ROUNDS = 50
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1
    ROUNDS_BETWEEN_VALIDATIONS = 2

    def __init__(self, global_model, host, port):

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.ready_client_sids = set()
        self.main_client_id = None
        self.model_id = str(uuid.uuid4())

        self.host = host
        self.port = port