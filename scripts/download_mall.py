"""Download the Mall dataset (88 MB) into data/mall_dataset.

The dataset is released for research use by its authors and is not
redistributed in this repository:

    K. Chen, C. C. Loy, S. Gong, T. Xiang. Feature Mining for Localised Crowd
    Counting. BMVC 2012.
    https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
"""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"

data = Path(__file__).resolve().parents[1] / "data"
data.mkdir(exist_ok=True)
target = data / "mall_dataset.zip"
if not (data / "mall_dataset" / "mall_gt.mat").exists():
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, target)
    with zipfile.ZipFile(target) as z:
        z.extractall(data)
print("ready:", data / "mall_dataset")
