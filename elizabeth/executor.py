import logging
from .data_class import (ProcessPipelineInfo,
                         ProcessPipelineOutput,
                         ModelPipelineOutput,
                         ModelPipelineInfo,
                         EvaluationPipelineOutput,
                         EvaluationPipelineInfo,
                         PipelineOutput,
                         PipelineInfo)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute_process_pipeline(ppi: ProcessPipelineInfo) -> ProcessPipelineOutput:
    logger.info("Process Pipeline Started")
    ppo = ProcessPipelineOutput()
    logger.info("Read Data Started")
    data = ppi.read_data(ppi.data_path)
    logger.info("Read Data Finished")
    if ppi.keep_data:
        ppo.data = data
    logger.info("Process Data Started")
    processed_data = ppi.process(data)
    logger.info("Process Data Finished")
    if ppi.keep_processed_data:
        ppo.processed_data = processed_data
    logger.info("Process Pipeline Finished")
    return ppo


def execute_model_pipeline(mpi: ModelPipelineInfo) -> ModelPipelineOutput:
    logger.info("Model Pipeline Started")
    mpo = ModelPipelineOutput()
    logger.info("Read Data Started")
    data = mpi.read_data(mpi.data_path)
    logger.info("Read Data Finished")
    if mpi.keep_data:
        mpo.data = data
    logger.info("PreProcess Data Started")
    processed_data = mpi.pre_process(data)
    logger.info("PreProcess Data Finished")
    if mpi.keep_processed_data:
        mpo.processed_data = processed_data
    logger.info("Train Model Started")
    model = mpi.train(processed_data)
    logger.info("Train Model Finished")
    logger.info("Save Model Started")
    mpi.save_model(model, mpi.model_output_path)
    logger.info("Save Model Finished")
    if mpi.keep_model:
        mpo.model = model
    logger.info("Model Pipeline Finished")
    return mpo


def execute_evaluation_pipeline(epi: EvaluationPipelineInfo) -> EvaluationPipelineOutput:
    logger.info("Evaluation Pipeline Started")
    epo = EvaluationPipelineOutput()
    logger.info("Read Data Started")
    data = epi.read_data(epi.data_path)
    logger.info("Read Data Finished")
    logger.info("Load Model Started")
    model = epi.load_model(epi.model_input_path)
    logger.info("Load Model Finished")
    if epi.keep_data:
        epo.data = data
    logger.info("PreProcess Data Started")
    processed_data = epi.pre_process(data, model)
    logger.info("PreProcess Data Finished")
    if epi.keep_processed_data:
        epo.processed_data = processed_data
    logger.info("Evaluate Model Started")
    evaluation_output = epi.evaluate(processed_data)
    logger.info("Evaluate Model Finished")
    if epi.keep_evaluation:
        epo.evaluation_output = evaluation_output
    logger.info("Evaluation Pipeline Finished")
    return epo


def execute_pipeline(pi: PipelineInfo) -> PipelineOutput:
    mpi = pi.model_pipeline_info
    epi = pi.evaluation_pipeline_info
    mpo = execute_model_pipeline(mpi)
    epo = execute_evaluation_pipeline(epi)
    return PipelineOutput(model_pipeline_output=mpo, evaluation_pipeline_output=epo)
