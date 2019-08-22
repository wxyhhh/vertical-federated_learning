
from secureprotol import PaillierEncrypt, FakeEncrypt
from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
import uuid


class FLServer(object):
    MIN_NUM_WORKERS = 1
    MAX_NUM_ROUNDS = 50
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1
    ROUNDS_BETWEEN_VALIDATIONS = 2
    KEY_LENGTH = 1024

    def __init__(self, global_model, host, port):

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.ready_client_sids = set()
        self.main_client_id = None
        self.model_id = str(uuid.uuid4())
        self.encrypt_operator = PaillierEncrypt()
        self.encrypt_operator.generate_key(self.KEY_LENGTH)
        self.public_key = self.encrypt_operator.get_public_key()

        self.host = host
        self.port = port

        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####

        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
                'min_train_size': 1200,
                'data_split': (0.6, 0.3, 0.1),  # train, test, valid
                'epoch_per_round': 1,
                'batch_size': 10
            })


        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid, data)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1 and self.main_client_id is not None:
                for rid in handle_client_ready:
                    emit('send_public_key', {
                        'model_id': self.model_id,
                        'round_number': -1,
                        'public_key': self.public_key,

                        'weights_format': 'pickle',
                        'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                    }, room=rid)
                emit('send_public_key', {
                    'model_id': self.model_id,
                    'round_number': -1,
                    'public_key': self.public_key,

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=self.main_client_id)
                self.train_next_round()

        @self.socketio.on('main_client_ready')
        def handle_main_ready(data):
            print('main_client ready for training', request.sid, data)
            self.main_client_id = request.sid
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1 and self.main_client_id is not None:
                for rid in handle_client_ready:
                    emit('send_public_key', {
                        'model_id': self.model_id,
                        'round_number': -1,
                        'public_key':self.public_key,

                        'weights_format': 'pickle',
                        'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                    }, room=rid)
                emit('send_public_key', {
                    'model_id': self.model_id,
                    'round_number': -1,
                    'public_key': self.public_key,

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=self.main_client_id)
                self.train_next_round()

        @self.socketio.on('client_gradient')
        def handle_client_gradients(data):
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", request.sid)
            for x in data:
                if x != 'weights':
                    print(x, data[x])
            # data:
            #   weights
            #   train_size
            #   valid_size
            #   train_loss
            #   train_accuracy
            #   valid_loss?
            #   valid_accuracy?

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])

                # tolerate 30% unresponsive clients
                if len(self.current_round_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )

                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)

                    if 'valid_loss' in self.current_round_client_updates[0]:
                        aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )
                        print("aggr_valid_loss", aggr_valid_loss)
                        print("aggr_valid_accuracy", aggr_valid_accuracy)

                    if self.global_model.prev_train_loss is not None and \
                            (
                                    self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .01:
                        # converges
                        print("converges! starting test phase..")
                        self.stop_and_eval()
                        return

                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                );
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again


    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)

# def obj_to_pickle_string(x):
#     return codecs.encode(pickle.dumps(x), "base64").decode()
#     # return msgpack.packb(x, default=msgpack_numpy.encode)
#     # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO
#
# def pickle_string_to_obj(s):
#     return pickle.loads(codecs.decode(s.encode(), "base64"))
#     # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.

    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()