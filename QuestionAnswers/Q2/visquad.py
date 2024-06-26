import json
import datasets

logger = datasets.logging.get_logger(__name__)


class ViQuADConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViQuADConfig, self).__init__(**kwargs)


class ViQuAD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViQuADConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="UIT-ViQuAD2.0",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    "is_impossible": datasets.Value("bool"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "data/new_train.json"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "data/valid.json"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data:
                context = example["context"]
                qas = example["qas"]
                for qa in qas:
                    question = qa["question"]
                    answers = qa["answers"]
                    answer_starts = [answer["answer_start"] for answer in answers]
                    answer_texts = [answer["text"] for answer in answers]
                    is_impossible = qa.get("is_impossible", False)
                    yield qa["id"], {
                        "context": context,
                        "question": question,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answer_texts,
                        },
                        "id": qa["id"],
                        "is_impossible": is_impossible,
                    }