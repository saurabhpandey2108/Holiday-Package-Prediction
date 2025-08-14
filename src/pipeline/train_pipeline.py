from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.segmentation import CustomerSegmentationTrainer


def run_training_pipeline():
	ingestor = DataIngestion()
	train_path, test_path = ingestor.initiate_data_ingestion()

	transformer = DataTransformation()
	train_arr, test_arr, preproc_path = transformer.initiate_data_transformation(train_path, test_path)

	trainer = ModelTrainer()
	score = trainer.initiate_model_trainer(train_arr, test_arr)

	# Train segmentation to support targeted strategies
	seg = CustomerSegmentationTrainer()
	seg_meta = seg.train(train_path, test_path, preproc_path)
	return score, seg_meta


if __name__ == "__main__":
	result = run_training_pipeline()
	print(result)
