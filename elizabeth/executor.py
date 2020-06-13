from .data_class import (ModelPipelineOutput,
                         ModelPipelineInfo,
                         EvaluationPipelineOutput,
                         EvaluationPipelineInfo,
                         PipelineOutput,
                         PipelineInfo)


def execute_model_pipeline(mpi: ModelPipelineInfo) -> ModelPipelineOutput:
    mpo = ModelPipelineOutput()
    data = mpi.read_data(mpi.data_path)
    if mpi.keep_data:
        mpo.data = data
    processed_data = mpi.pre_process(data)
    if mpi.keep_processed_data:
        mpo.processed_data = processed_data
    model = mpi.train(processed_data)
    mpi.save_model(model, mpi.model_output_path)
    if mpi.keep_model:
        mpo.model = model
    return mpo


def execute_evaluation_pipeline(epi: EvaluationPipelineInfo) -> EvaluationPipelineOutput:
    epo = EvaluationPipelineOutput()
    data = epi.read_data(epi.data_path)
    model = epi.load_model(epi.model_input_path)
    if epi.keep_data:
        epo.data = data
    processed_data = epi.pre_process(data, model)
    if epi.keep_processed_data:
        epo.processed_data = processed_data
    evaluation_output = epi.evaluate(processed_data)
    if epi.keep_evaluation:
        epo.evaluation_output = evaluation_output
    return epo


def execute_pipeline(pi: PipelineInfo) -> PipelineOutput:
    mpi = pi.model_pipeline_info
    epi = pi.evaluation_pipeline_info
    mpo = execute_model_pipeline(mpi)
    epo = execute_evaluation_pipeline(epi)
    return PipelineOutput(model_pipeline_output=mpo, evaluation_pipeline_output=epo)
