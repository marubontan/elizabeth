import pandas as pd
import pickle
from elizabeth.data_class import ModelPipelineInfo, EvaluationPipelineInfo
from elizabeth.executor import execute_model_pipeline, execute_evaluation_pipeline
from .setting import FIXTURE_DATA_PATH, FIXTURE_MODEL_PATH, TEST_MODEL_OUTPUT_PATH


class FakeModel:
    @staticmethod
    def predict(x_1, x_2):
        return 1 * x_1 + 2 * x_2


def fake_read_data(path):
    data = pd.read_csv(path)
    return data


def fake_pre_process(data):
    return data


def fake_training(_):
    return FakeModel


def fake_save_model(model, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(model, f)


def test_execute_model_pipeline():
    mpi = ModelPipelineInfo(model_output_path=TEST_MODEL_OUTPUT_PATH,
                            data_path=FIXTURE_DATA_PATH,
                            read_data=fake_read_data,
                            pre_process=fake_pre_process,
                            train=fake_training,
                            save_model=fake_save_model,
                            keep_data=True,
                            keep_processed_data=True,
                            keep_model=True)
    actual_mpo = execute_model_pipeline(mpi)
    expected_data = fake_read_data(FIXTURE_DATA_PATH)
    expected_processed_data = expected_data
    expected_model = FakeModel
    assert actual_mpo.data.equals(expected_data)
    assert actual_mpo.processed_data.equals(expected_processed_data)
    assert actual_mpo.model == expected_model


def fake_load_model(data_path):
    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    return model


def fake_evaluation_pre_process(data, model):
    return data


def fake_evaluate(_):
    return {'acc': 0.9, 'recall': 0.8}


def test_execute_evaluation_pipeline():
    epi = EvaluationPipelineInfo(model_input_path=FIXTURE_MODEL_PATH,
                                 data_path=FIXTURE_DATA_PATH,
                                 read_data=fake_read_data,
                                 load_model=fake_load_model,
                                 pre_process=fake_evaluation_pre_process,
                                 evaluate=fake_evaluate)
    epo = execute_evaluation_pipeline(epi)
    expected_data = fake_read_data(FIXTURE_DATA_PATH)
    expected_processed_data = expected_data
    expected_evaluation_output = {'acc': 0.9, 'recall': 0.8}
    assert epo.data.equals(expected_data)
    assert epo.processed_data.equals(expected_processed_data)
    assert epo.evaluation_output == expected_evaluation_output
