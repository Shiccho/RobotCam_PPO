import cv2
import pathlib
from PIL import Image
from torchvision import transforms

class DataLoder():
    def __init__(self, data_dir, shape=320):
        input_dir = data_dir
        input_imgs = pathlib.Path(input_dir).glob('**/*.png')
        self.data = []
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(shape),
            transforms.ToTensor()
        ])
        for path in input_imgs:
            img_file_name = str(path)
            img = cv2.imread(img_file_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_mask = cv2.inRange(img_gray, 3, 258)
            self.data.append(transform(Image.fromarray(img_mask)))
        
if __name__ == '__main__':
    dataLoader = DataLoder("mask_imgs")
    print(dataLoader.data[0])
