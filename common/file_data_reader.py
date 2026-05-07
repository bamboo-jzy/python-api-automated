"""
文件数据读取器模块

该模块提供了一个统一的接口来安全地读取多种格式的数据文件（JSON, CSV, Excel, TOML）。
核心功能包括路径安全验证、统一的日志记录、详细的错误处理以及数据摘要生成。

作者：资深Python开发工程师
创建日期：2026-02-04
依赖：tomllib, pathlib, typing, pandas, common.log_config
"""

from pathlib import Path
from typing import Any, Union

import pandas as pd
import tomllib

from common.log_config import setup_logger

# 初始化日志记录器
logger = setup_logger()


class FileDataReader:
    """
    文件数据读取器类

    提供统一的API来安全地读取不同格式的数据文件。
    包括路径验证、错误处理和数据摘要生成功能。
    """

    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """
        验证并返回绝对路径，确保安全性

        该方法验证文件路径是否有效且位于项目根目录内，
        防止路径遍历攻击，并检查文件是否存在和是否为文件类型。

        Args:
            file_path (Union[str, Path]): 输入的文件路径（相对或绝对）

        Returns:
            Path: 验证后的绝对路径对象

        Raises:
            PermissionError: 如果路径超出项目根目录范围
            FileNotFoundError: 如果文件不存在
            ValueError: 如果路径不是文件而是目录
        """
        # 将输入路径转换为Path对象
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        # 获取项目根目录路径
        project_root = Path(__file__).resolve().parent.parent

        # 处理绝对路径和相对路径
        if path_obj.is_absolute():
            absolute_file_path = path_obj.resolve(strict=False)
        else:
            absolute_file_path = (project_root / path_obj).resolve(strict=False)

        # 检查路径是否在项目根目录范围内（防止路径遍历攻击）
        try:
            absolute_file_path.relative_to(project_root)
        except ValueError:
            error_msg = f"拒绝访问项目根目录外的路径: {absolute_file_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)

        # 检查文件是否存在
        if not absolute_file_path.exists():
            error_msg = f"文件不存在: {absolute_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 检查路径是否为文件而非目录
        if not absolute_file_path.is_file():
            error_msg = f"路径非文件: {absolute_file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return absolute_file_path

    def _summarize_data(self, data: Any) -> str:
        """
        生成安全的数据摘要（避免日志爆炸）

        为读取的数据生成简要描述，用于日志记录。

        Args:
            data (Any): 待摘要的数据对象

        Returns:
            str: 数据摘要字符串
        """
        if isinstance(data, pd.DataFrame):
            # 处理DataFrame类型数据
            if data.empty:
                return "空数据集"
            # 只显示前3列以避免日志过长
            cols = list(data.columns[:3])
            col_repr = f"{cols}{'...' if len(data.columns) > 3 else ''}"
            return f"形状{data.shape} | 列预览: {col_repr} (共{len(data.columns)}列)"
        elif isinstance(data, dict):
            # 处理字典类型数据
            if not data:
                return "空字典"
            # 只显示前3个键以避免日志过长
            keys = list(data.keys())[:3]
            key_repr = f"{keys}{'...' if len(data) > 3 else ''}"
            return f"键数量: {len(data)} | 预览: {key_repr}"
        # 其他类型数据返回类型名
        return f"类型: {type(data).__name__}"

    def read_toml(self, file_path: Union[str, Path], **kwargs) -> dict[str, Any]:
        """
        读取TOML文件

        安全地读取并解析TOML格式的配置文件，包含错误处理和日志记录。

        Args:
            file_path (Union[str, Path]): TOML文件路径
            **kwargs: 传递给tomllib.load的额外参数，如自定义解析选项等

        Returns:
            dict[str, Any]: 解析后的字典数据，包含TOML文件的所有内容

        Raises:
            ValueError: 当文件不存在、解析失败或发生IO错误时抛出
            PermissionError: 当路径超出项目根目录范围时抛出
        """
        # 验证文件路径安全性
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取TOML文件: {absolute_file_path}")

        try:
            # 使用二进制模式打开文件并解析TOML内容
            with open(absolute_file_path, "rb") as f:
                data = tomllib.load(f, **kwargs)
            # 记录数据摘要以便调试
            logger.debug(f"TOML文件数据摘要：{self._summarize_data(data)}")
            return data
        except tomllib.TOMLDecodeError as e:
            # TOML解析错误处理
            error_msg = f"TOML 文件 '{absolute_file_path}' 解析失败: {str(e)[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except (OSError, IOError) as e:
            # 文件IO错误处理
            error_msg = f"TOML 文件 '{absolute_file_path}' IO错误: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        读取CSV文件

        安全地读取CSV格式的数据文件，返回pandas DataFrame对象。

        Args:
            file_path (Union[str, Path]): CSV文件路径
            **kwargs: 传递给pandas.read_csv的额外参数，如分隔符、编码等

        Returns:
            pd.DataFrame: 包含CSV数据的DataFrame对象

        Raises:
            ValueError: 当文件不存在、解析失败或发生IO错误时抛出
            PermissionError: 当路径超出项目根目录范围时抛出
        """
        # 验证文件路径安全性
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取CSV文件: {absolute_file_path}")

        try:
            # 使用pandas读取CSV文件
            df = pd.read_csv(absolute_file_path, **kwargs)
            # 记录数据摘要以便调试
            logger.debug(f"CSV文件数据摘要：{self._summarize_data(df)}")
        except pd.errors.EmptyDataError:
            # 空文件处理
            logger.warning(f"CSV 文件为空: {absolute_file_path}")
            df = pd.DataFrame()
        except (pd.errors.ParserError, OSError, IOError, ValueError) as e:
            # 解析错误或其他错误处理
            error_msg = f"CSV 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        return df

    def read_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        读取Excel文件

        安全地读取Excel格式的数据文件（.xlsx, .xls），返回pandas DataFrame对象。

        Args:
            file_path (Union[str, Path]): Excel文件路径
            **kwargs: 传递给pandas.read_excel的额外参数，如工作表索引、列名等

        Returns:
            pd.DataFrame: 包含Excel数据的DataFrame对象

        Raises:
            ValueError: 当文件不存在、解析失败或发生IO错误时抛出
            PermissionError: 当路径超出项目根目录范围时抛出
        """
        # 验证文件路径安全性
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取Excel文件: {absolute_file_path}")

        try:
            # 使用pandas读取Excel文件
            df = pd.read_excel(absolute_file_path, **kwargs)
            # 记录数据摘要以便调试
            logger.debug(f"Excel文件数据摘要：{self._summarize_data(df)}")
            return df
        except (ValueError, OSError, IOError) as e:
            # 错误处理
            error_msg = (
                f"Excel 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def read_json(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        读取JSON文件

        安全地读取JSON格式的数据文件，返回pandas DataFrame对象。

        Args:
            file_path (Union[str, Path]): JSON文件路径
            **kwargs: 传递给pandas.read_json的额外参数，如日期格式、数据类型等

        Returns:
            pd.DataFrame: 包含JSON数据的DataFrame对象

        Raises:
            ValueError: 当文件不存在、解析失败或发生IO错误时抛出
            PermissionError: 当路径超出项目根目录范围时抛出
        """
        # 验证文件路径安全性
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取JSON文件: {absolute_file_path}")

        try:
            # 使用pandas读取JSON文件
            df = pd.read_json(absolute_file_path, **kwargs)
            # 记录数据摘要以便调试
            logger.debug(f"JSON 文件数据摘要：{self._summarize_data(df)}")
            return df
        except (ValueError, OSError, IOError) as e:
            # 错误处理
            error_msg = (
                f"JSON 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e


if __name__ == "__main__":
    # 主程序入口点 - 示例用法
    reader = FileDataReader()
    res = reader.read_json(file_path="data/test.json")
    print(res)
