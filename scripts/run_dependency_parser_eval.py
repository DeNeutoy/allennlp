import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import lazy_groups_of

def main(data: str) -> None:
    """

    """
    data_files = glob.glob(os.path.join(data, "news.en-*"))

    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    predictor = Predictor.from_archive(archive, "biaffine-dependency-parser")

    batch_size = 64

    for data_file in data_files:
        data = (x.strip().split(" ") for x in open(data_file))
        for batch in lazy_groups_of(data, batch_size):
            json_batch = [{"sentence": x} for x in batch]
            predictor.predict_batch_json(json_batch)

    print(predictor.model.get_metrics())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a dependency parser on lots of data.")
    parser.add_argument('--data', type=str, help='The path to the data directory.')
    args = parser.parse_args()
    main(args.data)
