from tensorflow.python.client import device_lib

def get_available_gpus():
    r"""
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    r"""
    Returns the number of CPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']
