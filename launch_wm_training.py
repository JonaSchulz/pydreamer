import argparse
import json
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process
from typing import List

import generator
import train_wm_only
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, print_once, read_yamls)


def launch():
    os.environ['MLFLOW_RUN_ID'] = 'e70369e87890439b8ad5950235b18711'
    configure_logging('[launcher]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    # Mlflow

    worker_type, worker_index = get_worker_info()
    is_main_worker = worker_type is None or worker_type == 'learner'
    mlrun = mlflow_init(wait_for_resume=not is_main_worker)
    artifact_uri = mlrun.info.artifact_uri
    mlflow_log_params(vars(conf))

    subprocesses: List[Process] = []

    # Launch learner

    if belongs_to_worker('learner', 0):
        info('Launching learner')
        p = launch_learner(conf)
        subprocesses.append(p)

    # Wait & watch

    try:
        while len(subprocesses) > 0:
            check_subprocesses(subprocesses)
            time.sleep(1)
    finally:
        for p in subprocesses:
            p.kill()  # Non-daemon processes (learner) need to be killed


def launch_learner(conf):
    p = Process(target=train_wm_only.run, daemon=False, args=[conf])
    p.start()
    return p


def check_subprocesses(subprocesses):
    subp_finished = []
    for p in subprocesses:
        if not p.is_alive():
            if p.exitcode == 0:
                subp_finished.append(p)
                info(f'Generator process {p.pid} finished')
            else:
                raise Exception(f'Generator process {p.pid} died with exitcode {p.exitcode}')
    for p in subp_finished:
        subprocesses.remove(p)


def belongs_to_worker(work_type, work_index):
    """
    In case of distributed workers, checks if this work should execute on this worker.
    If not distributed, return True.
    """
    worker_type, worker_index = get_worker_info()
    return (
        (worker_type is None or worker_type == work_type) and
        (worker_index is None or worker_index == work_index)
    )


def get_worker_info():
    worker_type = None
    worker_index = None

    if 'TF_CONFIG' in os.environ:
        # TF_CONFIG indicates Google Vertex AI run
        tf_config = json.loads(os.environ['TF_CONFIG'])
        print_once('TF_CONFIG is set:', tf_config)
        if tf_config['cluster'].get('worker'):
            # If there are workers in the cluster, then it's a distributed run
            worker_type = {
                'chief': 'learner',
                'worker': 'generator',
            }[str(tf_config['task']['type'])]
            worker_index = int(tf_config['task']['index'])
            print_once('Distributed run detected, current worker is:', f'{worker_type} ({worker_index})')

    return worker_type, worker_index


if __name__ == '__main__':
    launch()
