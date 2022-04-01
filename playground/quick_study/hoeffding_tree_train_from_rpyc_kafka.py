import rpyc
import json
import time
import pandas as pd

from river import ensemble
from river import tree

if __name__ == '__main__':

    conn = rpyc.connect('localhost', 12345, config={'allow_public_attrs': True, 'allow_pickle': True})

    kafka_consumer = getattr(conn.root, 'get_consumer')()

    model = ensemble.AdaBoostClassifier(
        model=(
            tree.HoeffdingAdaptiveTreeClassifier(
                max_depth=3,
                split_criterion='gini',
                split_confidence=1e-2,
                grace_period=10,
                seed=0
            )
        ),
        n_models=10,
        seed=42
    )

    trained_event_count = 0

    while True:

        result = kafka_consumer.poll(timeout_ms=10000, max_records=None, update_offsets=True)

        print(len(result))
        for key, value in result.items():
            # fetch kafka message from kafka consumer via rpyc
            # take time to network transfer and unmarshal
            for record in value:
                receive_data = record.value

                row = pd.read_json(json.dumps(dict(receive_data)), typ='series', orient='records')
                y = row.pop('Y')

                # time to learn one event by 1 model
                start_time = time.time()
                model.learn_one(row, y)
                end_time = time.time()

                trained_event_count += 1

                print('\r#{} Events Trained, learn_one time spend: {} millisecond'.format(trained_event_count,
                                                                                          end_time - start_time), end='',
                      flush=True)

        time.sleep(1)

        # try:
        #     for msg in kafka_consumer:
        #
        #         # fetch kafka message from kafka consumer via rpyc
        #         # take time to network transfer and unmarshal
        #         receive_data = msg.value
        #
        #         row = pd.read_json(json.dumps(dict(receive_data)), typ='series', orient='records')
        #         y = row.pop('Y')
        #
        #         # time to learn one event by 1 model
        #         start_time = time.time()
        #         model.learn_one(row, y)
        #         end_time = time.time()
        #
        #         trained_event_count += 1
        #
        #         print('\r#{} Events Trained, learn_one time spend: {} millisecond'.format(trained_event_count,
        #                                                                                   end_time - start_time), end='',
        #               flush=True)
        # except TimeoutError:
        #     print("send a new request to rpyc server")
