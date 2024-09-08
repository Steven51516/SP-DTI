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