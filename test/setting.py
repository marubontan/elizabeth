import os

TEST_DIR_PATH = os.path.dirname(__file__)
TEST_OUTPUT_PATH = os.path.join(TEST_DIR_PATH, 'output')
TEST_MODEL_OUTPUT_PATH = os.path.join(TEST_OUTPUT_PATH, 'model.p')
FIXTURE_DIR_PATH = os.path.join(TEST_DIR_PATH, 'fixture')
FIXTURE_DATA_PATH = os.path.join(FIXTURE_DIR_PATH, 'data.csv')
FIXTURE_MODEL_PATH = os.path.join(FIXTURE_DIR_PATH, 'model.p')
