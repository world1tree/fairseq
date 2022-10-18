# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent
from hydra.core.config_store import ConfigStore

from .fairseq_task import FairseqTask, LegacyFairseqTask  # noqa


# register dataclass
TASK_DATACLASS_REGISTRY = {} # key: task_name, value: task_dataclass(继承FairseqDataClass)
TASK_REGISTRY = {}    # key: task_name, value: cls
TASK_CLASS_NAMES = set() # task_class_name


def setup_task(cfg: FairseqDataclass, **kwargs):
    task = None
    task_name = getattr(cfg, "task", None)

    if isinstance(task_name, str):
        # legacy tasks
        task = TASK_REGISTRY[task_name]
        if task_name in TASK_DATACLASS_REGISTRY:
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = dc.from_namespace(cfg)
    else:
        task_name = getattr(cfg, "_name", None)

        if task_name and task_name in TASK_DATACLASS_REGISTRY:
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = merge_with_parent(dc(), cfg)
            task = TASK_REGISTRY[task_name]

    assert (
        task is not None
    ), f"Could not infer task type from {cfg}. Available argparse tasks: {TASK_REGISTRY.keys()}. Available hydra tasks: {TASK_DATACLASS_REGISTRY.keys()}"

    return task.setup_task(cfg, **kwargs)


def register_task(name, dataclass=None):
    # 这个装饰器用来定义新的task
    # task必须继承FairseqTask, FairSeqTask应该是fairseq的核心
    # 注意!!!装饰器在定义处立即执行!!!
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        # 装饰类
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, FairseqTask):
            raise ValueError(
                "Task ({}: {}) must extend FairseqTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        # 可以引入一些额外的配置
        cls.__dataclass = dataclass
        if dataclass is not None:
            TASK_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            # 这里指定了配置对应的task name
            node._name = name
            cs.store(name=name, group="task", node=node, provider="fairseq")

        return cls

    return register_task_cls


def get_task(name):
    return TASK_REGISTRY[name]


def import_tasks(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path)) # 如果path是文件夹怎么办?
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            # task_name必须唯一
            # 在import_module的过程中，TASK_REGISTRY会被填充
            # key是task_name, value是不同task的具体实现类
            importlib.import_module(namespace + "." + task_name)

            # expose `task_parser` for sphinx
            if task_name in TASK_REGISTRY:
                # 下面是专门属于某个task的配置参数
                parser = argparse.ArgumentParser(add_help=False) # 此处task不显示help
                group_task = parser.add_argument_group("Task name")
                # fmt: off
                group_task.add_argument('--task', metavar=task_name,
                                        help='Enable this task with: ``--task=' + task_name + '``')
                # fmt: on
                group_args = parser.add_argument_group(
                    "Additional command-line arguments"
                )
                # 不同task的具体实现类继承了FairseqTask, 而FairSeqTask有add_args类方法可以
                # 为传入的parser增加额外的参数, 具体实现类需要有__dataclass属性, _dataclass可以在
                # register_task中传入，add_args会把dataclass中的所有配置应用到group_args上
                TASK_REGISTRY[task_name].add_args(group_args)
                # 把该task的配置参数加入全局变量
                globals()[task_name + "_parser"] = parser


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "fairseq.tasks")
