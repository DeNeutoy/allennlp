

from allennlp.data.dataset_readers.wic import WicReader
from allennlp.common.testing import AllenNlpTestCase

class TestWicReader(AllenNlpTestCase):


    def test_read(self):

        path = self.FIXTURES_ROOT / "data" / "wic"
        print(path)
        reader = WicReader(bert_model_name="bert-base-uncased")
        data = list(reader.read(path))

        for x in data:
            print(x)