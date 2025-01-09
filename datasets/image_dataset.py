import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path, PurePath
from torch.utils.data import Dataset

train_compose_list = [
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
test_compose_list = [
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_type: str = "resnet",  # "resnet", "vit"
        image_size: int = 384,
        **kwargs,
    ):
        self.df = df
        self.kwargs = kwargs
        self.is_augment = kwargs.get("is_augment", None)
        self.image_size = (image_size, image_size)

        if self.is_augment:
            if model_type == "resnet":
                self.transform = transforms.Compose(train_compose_list)
            elif model_type == "vit":
                train_compose_list.insert(0, transforms.Resize(self.image_size))
                self.transform = transforms.Compose(train_compose_list)
        else:
            if model_type == "resnet":
                self.transform = transforms.Compose(test_compose_list)
            elif model_type == "vit":
                test_compose_list.insert(0, transforms.Resize(self.image_size))
                self.transform = transforms.Compose(test_compose_list)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> None:
        row = self.df.iloc[index, :]
        # original_image_path = Path(row["img_path"])
        original_image_path = PurePath(f"./data/{row['img_path']}")
        original_image = Image.open(original_image_path)
        original_image_tensor = self.transform(original_image)

        label_id = row["label_id"]

        return original_image_tensor, label_id


if __name__ == "__main__":
    meta_file_path = Path("./data/time_slices_50.csv")
    df = pd.read_csv(meta_file_path)
    dataset = ImageDataset(df, model_type="resnet", is_augment=True)
    img_tensor, label = dataset[100]
    print(img_tensor.shape, label)
    print(dataset.df.shape)
    print(dataset.transform)
    print("Done")
