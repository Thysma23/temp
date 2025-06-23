from pathlib import Path

DISPLAY_CLASS_HELP = True

DATASET_PATH = Path("data/chest_Xray/")

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

RESIZE_DIM = (128, 128) # You can up this to (256, 256) if you want to use the original size

IS_DATA_AUGMENTED = True # Set to True if you want to use data augmentation (e.g. rotation, zoom, etc.) (recommended but use more performance)
