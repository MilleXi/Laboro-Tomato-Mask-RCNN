from data_preparation import process_dataset
from data_preprocess import TomatoDataset, visualize
train_data, test_data = process_dataset("laboro-tomato-DatasetNinja")
dataset = TomatoDataset(train_data, train=True)
image, target = dataset[0]
visualize(image, target)