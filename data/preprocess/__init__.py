from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .operators import *


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    ops = []
    for operator in op_param_list:
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]

        op = eval(op_name)(**param)
        ops.append(op)

    return ops
