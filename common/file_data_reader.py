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

logger = setup_logger()

class FileDataReader:
    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        project_root = Path(__file__).resolve().parent.parent

        if path_obj.is_absolute():
            absolute_file_path = path_obj.resolve(strict=False)
        else:
            absolute_file_path = (project_root / path_obj).resolve(strict=False)

        try:
            absolute_file_path.relative_to(project_root)
        except ValueError:
            error_msg = f"拒绝访问项目根目录外的路径: {absolute_file_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)

        if not absolute_file_path.exists():
            error_msg = f"文件不存在: {absolute_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not absolute_file_path.is_file():
            error_msg = f"路径非文件: {absolute_file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return absolute_file_path

    def _summarize_data(self, data: Any) -> str:
        """生成安全的数据摘要（避免日志爆炸）

        为读取的数据生成简要描述，用于日志记录。

        Args:
            data (Any): 待摘要的数据对象

        Returns:
            str: 数据摘要字符串
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return "空数据集"
            cols = list(data.columns[:3])
            col_repr = f"{cols}{'...' if len(data.columns) > 3 else ''}"
            return f"形状{data.shape} | 列预览: {col_repr} (共{len(data.columns)}列)"
        elif isinstance(data, dict):
            if not data:
                return "空字典"
            keys = list(data.keys())[:3]
            key_repr = f"{keys}{'...' if len(data) > 3 else ''}"
            return f"键数量: {len(data)} | 预览: {key_repr}"
        return f"类型: {type(data).__name__}"

    def read_toml(self, file_path: Union[str, Path], **kwargs) -> dict[str, Any]:
        """读取TOML文件

        Args:
            **kwargs: 传递给tomllib.load的额外参数

        Returns:
            dict[str, Any]: 解析后的字典数据

        Raises:
            ValueError: 当解析失败或IO错误时
        """
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取TOML文件: {absolute_file_path}")
        try:
            with open(absolute_file_path, "rb") as f:
                data = tomllib.load(f, **kwargs)
            logger.debug(f"TOML文件数据摘要：{self._summarize_data(data)}")
            return data
        except tomllib.TOMLDecodeError as e:
            error_msg = f"TOML 文件 '{absolute_file_path}' 解析失败: {str(e)[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except (OSError, IOError) as e:
            error_msg = f"TOML 文件 '{absolute_file_path}' IO错误: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取CSV文件: {absolute_file_path}")
        try:
            df = pd.read_csv(absolute_file_path, **kwargs)
            logger.debug(f"CSV文件数据摘要：{self._summarize_data(df)}")
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV 文件为空: {absolute_file_path}")
            df = pd.DataFrame()
        except (pd.errors.ParserError, OSError, IOError, ValueError) as e:
            error_msg = f"CSV 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        return df

    def read_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取Excel文件: {absolute_file_path}")
        try:
            df = pd.read_excel(absolute_file_path, **kwargs)
            logger.debug(f"Excel文件数据摘要：{self._summarize_data(df)}")
            return df
        except (ValueError, OSError, IOError) as e:
            error_msg = (
                f"Excel 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def read_json(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        absolute_file_path = self._validate_path(file_path)
        logger.debug(f"读取JSON文件: {absolute_file_path}")
        try:
            df = pd.read_json(absolute_file_path, **kwargs)
            logger.debug(f"JSON 文件数据摘要：{self._summarize_data(df)}")
            return df
        except (ValueError, OSError, IOError) as e:
            error_msg = (
                f"JSON 文件 '{absolute_file_path}' 读取/解析错误: {str(e)[:200]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e


if __name__ == "__main__":
    reader = FileDataReader()
    res = reader.read_json(file_path="data/test.json")
    print(res)
