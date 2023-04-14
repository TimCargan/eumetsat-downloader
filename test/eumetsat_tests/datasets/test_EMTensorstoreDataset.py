from absl import flags
from absl.testing import parameterized

from eumetsat.datasets.tensorstore_dataset import EMTensorstoreDataset
from hemera import path_translator

FLAGS = flags.FLAGS



class TestEMTesnorstoreDataset(parameterized.TestCase):

    def test_extract(self):
        path = path_translator.get_path("data") / "EUMETSAT" / "UK-EXT" / "img_z=1514764800,e=1515110340,f=3600.ts"
        ext = EMTensorstoreDataset(path)