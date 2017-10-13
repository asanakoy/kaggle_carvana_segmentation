from .abstract_dataset import ReadingDataset


class H5LikeFileInterface:
    def __init__(self, dataset: ReadingDataset):
        self.dataset = dataset
        self.current_kind = None

    def __getitem__(self, item):
        if isinstance(item, str):
            self.current_kind = item
            return self
        elif isinstance(item, int):
            idx = item
            s = None
        elif isinstance(item, tuple):
            idx = item[0]
            s = item[1:]
        else:
            raise Exception()

        if self.current_kind == 'images':
            data = self.dataset.get_image(idx)
        elif self.current_kind == 'masks':
            data = self.dataset.get_mask(idx)
        elif self.current_kind == 'names':
            data = self.dataset.im_names[idx]
        elif self.current_kind == 'alphas':
            data = self.dataset.get_alpha(idx)
        else:
            raise Exception()

        return data[s] if s is not None else data

    def __contains__(self, item):
        return item in ['images', 'masks', 'names']

    def __len__(self):
        return len(self.dataset)