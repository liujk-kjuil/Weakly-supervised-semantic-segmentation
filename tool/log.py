import sys
import os
import logging


class MyLogger:
    def __init__(self, log_root="logs/stage1", stdout_file="stdout.txt", stderr_file="stderr.txt"):
        # 创建递增的日志目录
        log_dir = self._create_exp_dir(log_root)

        # 保存日志路径
        self.log_dir = log_dir
        
        # 设置标准输出日志
        self.stdout_logger = self._setup_logger(
            name="stdout_logger",
            log_file=os.path.join(log_dir, stdout_file),
            stream=sys.stdout
        )
        
        # 设置错误输出日志
        self.stderr_logger = self._setup_logger(
            name="stderr_logger",
            log_file=os.path.join(log_dir, stderr_file),
            stream=sys.stderr
        )
        
        # 重定向标准输出和错误输出
        sys.stdout = self.StreamWrapper(self.stdout_logger)
        sys.stderr = self.StreamWrapper(self.stderr_logger)

    def _setup_logger(self, name, log_file, stream):
        """设置日志记录器，将日志保存到文件和流（终端）"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # 文件日志处理器
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 流日志处理器
        stream_handler = logging.StreamHandler(stream)
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def _create_exp_dir(self, log_root):
        """创建递增编号的日志目录"""
        os.makedirs(log_root, exist_ok=True)
        existing_dirs = [d for d in os.listdir(log_root) if d.startswith("exp") and d[3:].isdigit()]
        existing_dirs.sort(key=lambda x: int(x[3:]))  # 按数字部分排序

        # 获取下一个实验编号
        next_exp_num = 1
        if existing_dirs:
            next_exp_num = int(existing_dirs[-1][3:]) + 1

        # 创建目录并返回路径
        log_dir = os.path.join(log_root, f"exp{next_exp_num:02d}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    class StreamWrapper:
        """包装器将打印信息重定向到日志"""
        def __init__(self, logger):
            self.logger = logger

        def write(self, message):
            if message.strip():  # 跳过空消息
                self.logger.info(message.strip())

        def flush(self):
            pass  # 日志不需要显式刷新
