from collections import namedtuple

Config = namedtuple("Config", [
    "dataset_path",
    "models_dir",
    "folder",
    "img_rows",
    "img_cols",
    "target_rows",
    "target_cols",
    "num_channels",
    "network",
    "loss",
    "lr",
    "optimizer",
    "batch_size",
    "epoch_size",
    "use_clahe",
    "nb_epoch",
    "cycle_start_epoch",
    "predict_batch_size",
    "use_crop",
    "use_resize",
    "dbg",
    "save_images",
    "test_pad",
    "train_pad",
    "fold" # pass in argparse
])


