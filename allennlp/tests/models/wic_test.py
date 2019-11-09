from typing import Dict

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from _pytest.monkeypatch import MonkeyPatch

from allennlp.common.testing import ModelTestCase
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


class TestWiCModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.monkeypatch = MonkeyPatch()

        # monkeypatch the PretrainedBertModel to return the tiny test fixture model
        config_path = self.FIXTURES_ROOT / "bert" / "config.json"
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        config = BertConfig(str(config_path))
        self.monkeypatch.setattr(BertModel, "from_pretrained", lambda _: BertModel(config))
        self.monkeypatch.setattr(
            BertTokenizer, "from_pretrained", lambda _: BertTokenizer(vocab_path)
        )

    def tearDown(self):
        self.monkeypatch.undo()
        self.monkeypatch.undo()
        super().tearDown()

    def test_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "wic" / "wic.jsonnet"

        self.set_up_model(param_file, self.FIXTURES_ROOT / "data" / "wic")
        self.ensure_model_can_train_save_and_load(param_file)
