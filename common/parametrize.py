# -*- coding: utf-8 -*-
"""
pytest参数化数据加载模块

模块功能：提供基于文件的pytest参数化装饰器工具，支持从外部文件读取测试数据，
          自动转换为pytest.mark.parametrize所需格式，支持自定义标记及自动生成用例ID。
适用场景：自动化测试场景中，需从Excel/CSV等文件批量加载测试数据，
          并为不同测试用例配置自定义pytest标记（如skip、xfail等）的场景。
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import pytest
from pandas import DataFrame

from common.file_data_reader import FileDataReader
from common.log_config import setup_logger

logger = setup_logger()


def _parse_marks(val: Any) -> List[str]:
    """解析mark列中的标记，将其转换为标记列表"""
    if not val or (isinstance(val, float) and str(val) == "nan"):
        return []
    raw_str = str(val).strip()
    if not raw_str:
        return []
    return [m.strip() for m in raw_str.split("-") if m.strip()]


def _dataframe_to_parametrize_data(
    df: DataFrame,
) -> Tuple[str, List[tuple], Dict[int, List[str]]]:
    """
    将DataFrame转换为pytest参数化所需的格式

    Args:
        df: 包含测试数据的DataFrame

    Returns:
        包含参数名、参数数据和标记映射的元组
    """
    mark_col = df.get("mark")
    mark_data: Dict[int, List[str]] = {}

    if mark_col is not None:
        # 处理mark列：解析标记并创建映射
        mark_series = mark_col.apply(_parse_marks)
        mark_data = {
            idx: marks for idx, marks in mark_series.to_dict().items() if marks
        }

    # 移除mark列并处理空值
    non_mark_df = df.drop(columns=["mark"], errors="ignore").fillna("")

    # 确保列名是字符串并去除首尾空格
    non_mark_df.columns = non_mark_df.columns.astype(str).str.strip()

    # 构建参数名字符串和参数数据列表
    parameterized_variables = ",".join(non_mark_df.columns)
    parameterized_data = [tuple(row) for row in non_mark_df.values.tolist()]

    return parameterized_variables, parameterized_data, mark_data


def _build_param_objects(
    data_list: List[tuple], marks_mapping: Dict[int, List[str]]
) -> List[Any]:
    """
    构建带有标记和自定义ID的 pytest.param 对象列表。

    Args:
        data_list: 参数数据列表
        marks_mapping: 标记映射字典，键为数据索引，值为标记列表

    Returns:
        包含pytest.param对象的列表
    """
    built_params = []

    for index, item in enumerate(data_list):
        current_marks = []

        # 如果当前索引有对应的标记，则添加到current_marks中
        if index in marks_mapping:
            for mark_name in marks_mapping[index]:
                if hasattr(pytest.mark, mark_name):
                    current_mark = getattr(pytest.mark, mark_name)
                    current_marks.append(current_mark)
                else:
                    logger.warning(
                        f"行索引 [{index}] 中的标记 '{mark_name}' 未注册到 pytest，将被忽略。"
                    )

        # 使用更描述性的case_id格式
        case_id = f"case_{index:03d}"

        param_obj = pytest.param(*item, marks=current_marks, id=case_id)
        built_params.append(param_obj)

    return built_params


def parametrize(
    file_path: Union[str, Path], **kwargs
) -> Callable[[Callable], Callable]:
    """
    pytest参数化装饰器，支持从外部文件加载测试数据

    Args:
        file_path: 测试数据文件路径
        **kwargs: 传递给文件读取方法的额外参数

    Returns:
        pytest.mark.parametrize装饰器

    Raises:
        FileNotFoundError: 当指定的文件不存在时
        ValueError: 当文件中没有有效测试数据时
        TypeError: 当文件类型不受支持时
    """
    # 验证文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {file_path}")

    # 获取文件后缀
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    suffix = path_obj.suffix.lower()

    # 创建文件读取器并读取数据
    reader = FileDataReader()
    try:
        if suffix == ".csv":
            data_frame = reader.read_csv(path_obj, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            data_frame = reader.read_excel(path_obj, **kwargs)
        else:
            raise TypeError(f"不支持的文件类型: {suffix}")
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        raise

    # 验证数据有效性
    if data_frame is None or data_frame.empty:
        error_msg = f"{file_path} 文件中无有效测试数据"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 转换数据格式
    variables, data, marks_map = _dataframe_to_parametrize_data(
        cast(DataFrame, data_frame)
    )

    # 构建参数对象
    final_data = _build_param_objects(data, marks_map)

    # 返回参数化装饰器
    return pytest.mark.parametrize(variables, final_data)
