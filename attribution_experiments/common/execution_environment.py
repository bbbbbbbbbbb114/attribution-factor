"""Explicit backend selection; existing launches remain on udocker by default."""
import os


def backend_name():
    value = os.environ.get('EXPERIMENT_CONTAINER_BACKEND', 'udocker')
    if value not in ('udocker', 'remote_docker', 'local_docker'):
        raise ValueError(f'Unknown execution backend: {value}')
    return value


def ExecutionEnvironment(**kwargs):
    if backend_name() == 'local_docker':
        from .local_docker_environment import LocalDockerEnvironment
        return LocalDockerEnvironment(**kwargs)
    if backend_name() == 'remote_docker':
        from .remote_environment import RemoteDockerEnvironment
        return RemoteDockerEnvironment(**kwargs)
    from .udocker_environment import UdockerEnvironment
    return UdockerEnvironment(**kwargs)
