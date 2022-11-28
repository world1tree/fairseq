# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from typing import Union
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

REGISTRIES = {}


def setup_registry(registry_name: str, base_class=None, default=None, required=False):
    # 该函数用于**生成装饰器**, 装饰器用来在不同文件夹下的文件中进行register具体类
    # 生成的装饰器只能修饰继承base_class的类
    # 各个文件夹下面的__init__.py文件需要调用该函数
    # 为什么register_task不通过这种方式去做?
    assert registry_name.startswith("--")
    # registry_name类似命令行参数，如--criterion
    registry_name = registry_name[2:].replace("-", "_")

    REGISTRY = {}               # 与register_task类似, 存储name: cls
    REGISTRY_CLASS_NAMES = set() # 这个后面并没有传出去，只是为了防止出现重复的class name
    DATACLASS_REGISTRY = {}

    # maintain a registry of all registries(task除外)
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        "registry": REGISTRY,
        "default": default,
        "dataclass_registry": DATACLASS_REGISTRY,
    }

    def build_x(cfg: Union[DictConfig, str, Namespace], *extra_args, **extra_kwargs):
        if isinstance(cfg, DictConfig):
            choice = cfg._name

            if choice and choice in DATACLASS_REGISTRY:
                dc = DATACLASS_REGISTRY[choice]
                cfg = merge_with_parent(dc(), cfg)
        elif isinstance(cfg, str):
            choice = cfg
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice]()
        else:
            choice = getattr(cfg, registry_name, None)
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice].from_namespace(cfg)

        if choice is None:
            if required:
                raise ValueError("{} is required!".format(registry_name))
            return None

        cls = REGISTRY[choice]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls

        return builder(cfg, *extra_args, **extra_kwargs)

    # 第一层是装饰器需要的参数
    def register_x(name, dataclass=None):
        # 第二层是装饰器修饰的函数或类
        # 由于不需要知道类的参数，所以不需要第三层
        # dataclass必须继承自FairseqDataclass
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(registry_name, name)
                )
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    "Cannot register {} with duplicate class name ({})".format(
                        registry_name, cls.__name__
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError(
                    "{} must extend {}".format(cls.__name__, base_class.__name__)
                )

            if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
                raise ValueError(
                    "Dataclass {} must extend FairseqDataclass".format(dataclass)
                )

            cls.__dataclass = dataclass
            if cls.__dataclass is not None:
                DATACLASS_REGISTRY[name] = cls.__dataclass

                cs = ConfigStore.instance()
                node = dataclass()
                node._name = name
                cs.store(name=name, group=registry_name, node=node, provider="fairseq")

            REGISTRY[name] = cls

            # 注意这里没有实例化！！！
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY, DATACLASS_REGISTRY
