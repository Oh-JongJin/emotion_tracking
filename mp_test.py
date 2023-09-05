import time
import pandas as pd
from multiprocessing import Process


def save(data: pd.DataFrame):
    while True:
        epoch = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
        print(data)
        time.sleep(1)


if __name__ == '__main__':
    p1 = None
    csv = pd.DataFrame()

    epoch = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
    if p1 is None:
        p1 = Process(target=save, args=(epoch,))
        p1.start()
        print('p1 start')
