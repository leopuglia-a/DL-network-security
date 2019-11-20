import datetime


def to_bin(x):
    if x == 'Syn':
        return 1
    return 0


def date_str_to_ms(date_time_str):
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    return date_time_obj.timestamp()


batch_size = 128
epochs = 10
