# SP-DTI: Subpocket-Informed Transformer for Drug-Target Interaction Prediction

## Overview

<p align="center">
  <img src="images/SP-DTI.png" alt="SP-DTI Overview" width="600px" />
</p>

SP-DTI is a Subpocket-Informed Transformer model designed for predicting drug-target interactions (DTI). The model leverages both protein subpocket features and drug representations to enhance interaction prediction accuracy.

## Installation

### With FlexMol

Please visit the [FlexMol repository](https://github.com/Steven51516/FlexMol/) for instructions on installing FlexMol. FlexMol is a powerful and flexible toolkit designed to advance molecular relation learning (MRL) by enabling the construction and evaluation of diverse model architectures across various datasets and performance metrics. FlexMol streamlines the development process, reduces repetitive coding efforts, and ensures fair comparisons of different models.

### Build from Source

The original SP-DTI model was developed using FlexMol. We are currently working on a version that does not rely on FlexMol, and it will be made available in the SP-DTI folder once completed.


## Tutorials

The following code example demonstrates how to test the SP-DTI model using the DAVIS dataset with FlexMol.

```python
from FlexMol.dataset.loader import load_DAVIS
from FlexMol.encoder import FlexMol
from FlexMol.task import BinaryTrainer

train = load_DAVIS("data/DAVIS/train.txt")
val = load_DAVIS("data/DAVIS/val.txt")
test = load_DAVIS("data/DAVIS/test.txt")

FM = FlexMol()
drug_encoder = FM.init_drug_encoder("GCN_Chemberta")
protein_encoder1 = FM.init_prot_encoder("GCN_ESM", pdb=True, data_dir="data/DAVIS/pdb/")
protein_encoder2 = FM.init_prot_encoder("Subpocket", pdb=True, data_dir="data/DAVIS/pdb/")
combined_output = FM.set_interaction([drug_encoder, protein_encoder1, protein_encoder2], "SP-DTI") 
output = FM.apply_mlp(combined_output, head=1)

FM.build_model()

# Set up the trainer with specified parameters and metrics
trainer = BinaryTrainer(
    FM,
    task = "DTI",
    test_metrics=["accuracy", "precision", "recall", "f1"],
    device="cpu",
    early_stopping="roc-auc",
    epochs=30,
    patience=10,
    lr=0.0001,
    batch_size=32
)
# Prepare the datasets for training, validation, and testing
train_data, val_data, test_data = trainer.prepare_datasets(train_df=train, val_df=val, test_df=test)

trainer.train(train_data, val_data)

trainer.test(test_data)

```

