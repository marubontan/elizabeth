# elizabeth
## Overview
This is a machine learning pipeline library.  
About pre-processing, modeling and evaluation, this library supplies pipeline. By filling in the pipeline's demand such 
as data file path, load function, pre-processing function, training function and so on, you can compose executable pipeline.

## Concept
The main point and philosophy of this library is to restrict the freedom of developer for the clear view of machine leaning flow.  
When there are different demands between modeling and evaluation or some people need to work on the same machine leaning task, 
it can become very difficult to keep the machine leaning flow clear and concise. This library assigns some restrictions 
to compose the pipeline such that all the time the data or model should be loaded by explicitly given path.  
Also, you can explicitly keep and pass all the setting such as used function, data path to avoid confusion.  

## Usage
You can do three things, Modeling pipe, evaluation pipe and they both at once. The main usage is consistent. You fill in 
the demand the data class asks. And you can give the instance of it to the executor function.  

## Example
This is the example with linear regression.  

The code below is for model pipeline. Define necessary functions and data path for pipeline and compose `ModelPipelineInfo`.  
That object is given to the `execute_model_pipeline`.  

```python
import pickle
from sklearn.linear_model import LinearRegression
from elizabeth.executor import execute_model_pipeline
from elizabeth.data_class import ModelPipelineInfo


def read_data(data_path: str):
    data = pd.read_csv(data_path)
    return data

def pre_process(data: pd.DataFrame) -> pd.DataFrame:
    processed_data = data.copy()
    processed_data['x_1'] = processed_data['x_1'] / sum(processed_data['x_1'])
    processed_data['x_2'] = processed_data['x_2'] / sum(processed_data['x_2'])
    return processed_data

def train(data: pd.DataFrame):
    reg = LinearRegression().fit(data[['x_1', 'x_2']], data['y'])
    return reg

def save_model(model, save_to: str):
    with open(save_to, 'wb') as f:
        pickle.dump(model, f)


mpi_setting = {'model_output_path':'model.p',
               'data_path':'data.csv',
               'read_data':read_data,
               'pre_process':pre_process,
               'train':train,
               'save_model':save_model}
mpi = ModelPipelineInfo(**mpi_setting)

mpo = execute_model_pipeline(mpi)
```

As a default, `execute_model_pipeline` return the object which contains the information of data, processed data and
trained model. By argument, you can choose if you keep those information in the output object.  

Evaluation pipeline is almost same.  
```python
from sklearn.metrics import mean_squared_error
from elizabeth.executor import execute_evaluation_pipeline
from elizabeth.data_class import EvaluationPipelineInfo

model_input_path = 'model.p'
data_path = 'test_data.csv'

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluation_pre_process(data, model):
    pred = model.predict(data[['x_1', 'x_2']])
    return (data['y'], pred)

def evaluate(target_and_pred):
    target, pred = target_and_pred
    mse = mean_squared_error(target, pred)
    return {'mse': mse}

epi_setting = {'model_input_path': 'model.p',
               'data_path':'test_data.csv',
               'read_data':read_data,
               'load_model':load_model,
               'pre_process':evaluation_pre_process,
               'evaluate':evaluate}
epi = EvaluationPipelineInfo(**epi_setting)

epo = execute_evaluation_pipeline(epi)
```

If you want to make a set of modeling and evaluation, you can compose the `PipelineInfo` object and you can execute the 
modeling and evaluation end to end.  

```python
from elizabeth.data_class import PipelineInfo
from elizabeth.executor import execute_pipeline

pipe_info = PipelineInfo(model_pipeline_info=mpi, evaluation_pipeline_info=epi)
pipeline_outptu = execute_pipeline(pipe_info)
```

