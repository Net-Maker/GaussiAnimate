# Source: https://github.com/yaoyx689/RigGS
# Inherited reconstruction: https://github.com/CVMI-Lab/SC-GS
# Original upstream notices and terms are preserved in the adjacent licenses.

import pickle


def save_obj(path, obj):
    file = open(path, 'wb')
    obj_str = pickle.dumps(obj)
    file.write(obj_str)
    file.close()


def load_obj(path):
    file = open(path, 'rb')
    obj = pickle.loads(file.read())
    file.close()
    return obj
