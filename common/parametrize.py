# -*- coding: utf-8 -*-
"""
pytest参数化数据加载模块

模块功能：提供基于文件的pytest参数化装饰器工具，支持从外部文件读取测试数据，
          自动转换为pytest.mark.parametrize所需格式，支持自定义标记及自动生成用例ID。
适用场景：自动化测试场景中，需从Excel/CSV等文件批量加载测试数据，
          并为不同测试用例配置自定义pytest标记（如skip、xfail等）的场景。
"""

import os
from typing import Any, Callable, Dict, List, Tuple, cast

import pytest
from pandas import DataFrame

from common.file_data_reader import FileDataReader
from common.log_config import setup_logger

logger = setup_logger()


def _dataframe_to_parametrize_data(
    df: DataFrame,
) -> Tuple[str, List[tuple], Dict[int, List[str]]]:
    """
    将包含测试数据的DataFrame转换为pytest.mark.parametrize可直接使用的数据格式

    处理逻辑：
        1. 提取并解析 'mark' 列（若有），生成长列表映射。
        2. 移除 'mark' 列，填充空值为空字符串。
        3. 生成参数字符串和元组数据列表。

    Args:
        df (DataFrame): 包含测试数据的DataFrame对象。

    Returns:
        Tuple[str, List[tuple], Dict[int, List[str]]]:
            - 参数字符串 (e.g., "username,password")
            - 参数数据列表 (List[tuple])
            - 标记映射字典 (行索引 -> 标记名列表)
    """
    mark_col = df.get("mark")
    mark_data: Dict[int, List[str]] = {}

    if mark_col is not None:
        # 处理mark列：fillna -> 转字符串 -> 按 '-' 分割 -> 去除空白项 -> 去除每项的首尾空格
        def parse_marks(val: Any) -> List[str]:
            if not val or (isinstance(val, float) and str(val) == "nan"):
                return []
            raw_str = str(val).strip()
            if not raw_str:
                return []
            return [m.strip() for m in raw_str.split("-") if m.strip()]

        mark_series = mark_col.apply(parse_marks)
        mark_data = {
            idx: marks for idx, marks in mark_series.to_dict().items() if marks
        }

    non_mark_df = df.drop(columns=["mark"], errors="ignore").fillna("")

    non_mark_df.columns = non_mark_df.columns.astype(str).str.strip()

    parameterized_variables = ",".join(non_mark_df.columns)
    parameterized_data = [tuple(row) for row in non_mark_df.values.tolist()]

    return parameterized_variables, parameterized_data, mark_data


def parametrize(file_path: str, **kwargs) -> Callable[[Callable], Callable]:
    """
    从指定文件加载测试数据，生成带ID和自定义标记的pytest参数化装饰器。

    核心流程：
        1. 读取文件数据至DataFrame。
        2. 解析参数列、数据行及标记列。
        3. 构建 pytest.param 对象，自动绑定标记并生成 "case_{index}" 格式的ID。
        4. 返回 pytest.mark.parametrize 装饰器。

    Args:
        file_path (str): 测试数据文件路径（支持FileDataReader可读取的类型：Excel/CSV等）。
        **kwargs: 传递给 FileDataReader.read() 的额外参数（如 sheet_name, encoding, sep 等）。

    Returns:
        Callable[[Callable], Callable]: pytest的参数化装饰器函数。

    Raises:
        ValueError: 当文件中无有效测试数据时抛出。
        FileNotFoundError: 当指定的 file_path 文件不存在时。
        IOError: 当文件读取失败时。

    Note:
        1. 生成的测试用例ID格式为：`case_0`, `case_1`, ... `case_N`，便于在报告中排序和定位。
        2. 若 'mark' 列中的标记名称在 pytest.mark 中不存在，该标记会被忽略并记录警告。
        3. 数据文件中的空值会被统一替换为空字符串。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {file_path}")

    reader = FileDataReader(file_path)
    try:
        _, data_frame = reader.read(**kwargs)
        data_frame = cast(DataFrame, data_frame)
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        raise

    if data_frame is None or data_frame.empty:
        error_msg = f"{file_path} 文件中无有效测试数据"
        logger.error(error_msg)
        raise ValueError(error_msg)

    variables, data, marks_map = _dataframe_to_parametrize_data(cast(DataFrame, data_frame))

    def _build_param_objects(
        data_list: List[tuple], marks_mapping: Dict[int, List[str]]
    ) -> List[Any]:
        """
        构建带有标记和自定义ID的 pytest.param 对象列表。
        """
        built_params = []

        for index, item in enumerate(data_list):
            current_marks = []

            if index in marks_mapping:
                for mark_name in marks_mapping[index]:
                    if hasattr(pytest.mark, mark_name):
                        current_marks.append(getattr(pytest.mark, mark_name))
                    else:
                        logger.warning(
                            f"行索引 [{index}] 中的标记 '{mark_name}' 未注册到 pytest，将被忽略。"
                        )

            case_id = f"case_{index}"

            param_obj = pytest.param(*item, marks=current_marks, id=case_id)
            built_params.append(param_obj)

        return built_params

    final_data = _build_param_objects(data, marks_map)

    return pytest.mark.parametrize(variables, final_data)
