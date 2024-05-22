<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill DistilBERT Module

This repository contains the code supporting the DistilBERT target model for use with [Autodistill](https://github.com/autodistill/autodistill).

DistilBERT is a languae model architecture commonly used in training sentence classification models. You can use `autodistill` to train a DistilBERT model that classifies text.

## Installation

To use the DistilBERT target model, you will need to install the following dependency:

```bash
pip3 install autodistill-distilbert-text
```

## Quickstart

The DistilBERT module takes in `.jsonl` files and trains a text classification model.

Each record in the JSONL file should have an entry called `text` that contains the text to be classified. The `label` entry should contain the ground truth label for the text. This format is returned by Autodistill base text classification models like the GPTClassifier.

Here is an example entry of a record used to train a research paper subject classifier:

```json
{"title": "CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl", "content": "arXiv:2405.11039v1 Announce Type: new \nAbstract: The Common Crawl (CC) corpus....", "classification": "natural language processing"}
```

```python
from autodistill_distilbert import DistilBERT

target_model = DistilBERT()

# train a model
target_model.train("./data.jsonl", epochs=200)

# run inference on the new model
pred = target_model.predict("Geospatial data.", conf=0.01)

print(pred)
# geospatial
```

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
