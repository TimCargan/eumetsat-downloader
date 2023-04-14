from absl import flags
from absl.testing import parameterized

from eumetsat.datasets.utils import FileNameProps

FLAGS = flags.FLAGS


class test_path_translater(parameterized.TestCase):

    @parameterized.parameters(("img_z=123,e=456,f=789.ts", {"z": 123, "e": 456, "f": 789}),
                              ("random_prefix_z=123,e=456,f=789.ts", {"z": 123, "e": 456, "f": 789}),
                              ("img_z=123,e=456,f=789,s=36.gz.tfds", {"z": 123, "e": 456, "f": 789}),
                              ("img_f=789,z=123,s=36,e=456.gz.tfds", {"z": 123, "e": 456, "f": 789})
                               )

    def test_extract(self, test, true):
        ext = FileNameProps.from_str(test)

        self.assertEqual(ext.time_zero, true["z"])
        self.assertEqual(ext.time_end, true["e"])
        self.assertEqual(ext.freq, true["f"])
