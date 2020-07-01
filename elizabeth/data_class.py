from typing import Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class ProcessPipelineOutput:
    data: Optional[Any] = None
    processed_data: Optional[Any] = None


@dataclass
class ModelPipelineOutput:
    data: Optional[Any] = None
    processed_data: Optional[Any] = None
    model: Optional[Any] = None


@dataclass
class EvaluationPipelineOutput:
    data: Optional[Any] = None
    processed_data: Optional[Any] = None
    evaluation_output: Optional[Any] = None


@dataclass
class PipelineOutput:
    model_pipeline_output: Optional[ModelPipelineOutput] = None
    evaluation_pipeline_output: Optional[EvaluationPipelineOutput] = None


@dataclass
class ProcessPipelineInfo:
    data_path: str
    read_data: Callable
    process: Callable
    keep_data: bool = True
    keep_processed_data: bool = True


@dataclass
class ModelPipelineInfo:
    model_output_path: str
    data_path: str
    read_data: Callable
    pre_process: Callable
    train: Callable
    save_model: Callable[[Any, str], None]
    keep_data: bool = True
    keep_processed_data: bool = True
    keep_model: bool = True


@dataclass
class EvaluationPipelineInfo:
    model_input_path: str
    data_path: str
    read_data: Callable
    load_model: Callable[[str], Any]
    pre_process: Callable[[PipelineOutput, Any], Any]
    evaluate: Callable[[Any], Any]
    keep_data: bool = True
    keep_processed_data: bool = True
    keep_evaluation: bool = True


@dataclass
class PipelineInfo:
    model_pipeline_info: ModelPipelineInfo
    evaluation_pipeline_info: EvaluationPipelineInfo
