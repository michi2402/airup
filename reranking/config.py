#training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 3
LOGGING_STEPS = 20

#question enriching
AMOUNT_SAME_SIMILAR = 2
AMOUNT_MID_SIMILAR = 1
AMOUNT_FAR_SIMILAR = 1

#labels
LABEL_FAR_NEG = 0
LABEL_MID_NEG = 0
LABEL_CLOSE_NEG = 0

#paths etc
DATA_PATH = "../BioASQ-training13b/training13b.json"
DATA_FOLDER = "../BioASQ-training13b/"
RERANKER_PATH = "./reranker"
TEST_FILE_EXTENSION = "_test_dataset"
TRAIN_FILE_EXTENSION = "_train_dataset"

#models
CROSS_ENCODER_PRETRAINED_MODEL_MINILM_MARCO = "cross-encoder/ms-marco-MiniLM-L12-v2"
SENTENCE_TRANSFORMER_PRETRAINED_MODEL_MINILM = "all-MiniLM-L12-v2"
SENTENCE_TRANSFORMER_BIOBERT = "dmis-lab/biobert-v1.1"