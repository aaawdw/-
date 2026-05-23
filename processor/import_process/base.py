import logging
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

import colorlog

from processor.import_process.config import ImportConfig, get_config
from processor.import_process.exceptions import ImportProcessError

T = TypeVar("T")
class BaseNode(ABC):
    name:str='base_node'

    def __init__(self, config: Optional[ImportConfig]=None):

        self.config = config or get_config()
        self.logger=logging.getLogger(f'import.{self.name}')

    def __call__(self, state):
        try:
            self.logger.info(f'---{self.name} 开始 ---')
            result= self.process(state)
            self.logger.info(f'---{self.name} 完成 ---')
            return result
        except Exception as e:
            self.logger.error(f'{self.name}执行失败:{e}')
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    # 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置日志格式

    Args:
        level: 日志级别
    """

    logger = logging.getLogger()
    logger.setLevel(level)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',  # INFO 显示为绿色
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))

    logger.handlers.clear()
    logger.addHandler(handler)
