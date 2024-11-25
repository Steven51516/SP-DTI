import argparse
import os
import sys
import os
from FlexMol.dataset.loader import *
from FlexMol.encoder.FM import *
from FlexMol.task import *


esm_pickes_dir = None
chemberta_pickles_dir = None
subpocket_pickles_dir = None
device = 'cuda:0'
epoch = 30
patience = 7
lr = 0.0001
batch_size=64


def load_data(split_path, task = "davis"):
    """Load the dataset split from the specified path."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Dataset split not found at {split_path}")
    print(f"Loading data from {split_path}...")
    if task == "davis":
        return load_DAVIS(split_path)
    elif task == "biosnap":
        return load_BIOSNAP(split_path)
    return None

def main():
    parser = argparse.ArgumentParser(description="Run the training and evaluation pipeline.")
    
    parser.add_argument('--task', type=str, required=True, choices=['davis', 'biosnap'], 
                        help="Dataset to use (davis or biosnap)")
    parser.add_argument('--train_split', type=str, required=True, 
                        help="Path to the training dataset split")
    parser.add_argument('--val_split', type=str, required=True, 
                        help="Path to the validation dataset split")
    parser.add_argument('--test_split', type=str, required=True, 
                        help="Path to the testing dataset split")
    parser.add_argument('--pdb_dir', type=str, required=True, 
                        help="Path to the directory containing PDB files")
    parser.add_argument('--subpocket_dir', type=str, required=True, 
                        help="Path to the directory containing subpocket files")
    parser.add_argument('--metrics_dir', type=str, required=True, 
                        help="Path to save the metrics output")

    args = parser.parse_args()


    train = load_data(args.train_split, args.task).head(2)
    val = load_data(args.val_split, args.task).head(2)
    test = load_data(args.test_split, args.task).head(2)

    if not os.path.exists(args.pdb_dir):
        raise FileNotFoundError(f"PDB directory not found at {args.pdb_dir}")
    print(f"Using PDB files from {args.pdb_dir}...")

    if not os.path.exists(args.subpocket_dir):
        raise FileNotFoundError(f"Subpocket directory not found at {args.subpocket_dir}")
    print(f"Using subpocket files from {args.subpocket_dir}...")

    FM = FlexMol()
    de = FM.init_drug_encoder("GCN_Chemberta", output_feats = 128) 
    pe = FM.init_prot_encoder("GCN_ESM", pdb=True, data_dir = args.pdb_dir, pickle_dir = esm_pickes_dir, output_feats=128, hidden_feats=[128,128,128])
    subpocket = FM.init_prot_encoder("Subpocket", pdb=True, pdb_dir = args.pdb_dir, subpocket_dir = args.subpocket_dir,  pickle_dir = subpocket_pickles_dir)
    dp = FM.set_interaction([de, pe], "cat")
    output = FM.set_interaction([subpocket, dp], "pocket_attention")
    FM.build_model()
    trainer = BinaryTrainer(FM, task = "DTI", early_stopping="roc-auc", test_metrics=["roc-auc", "pr-auc", "precision", "recall",  "f1"], 
                            device=device, epochs=epoch, patience=patience, lr=lr, batch_size=batch_size, auto_threshold = "max-f1", metrics_dir = args.metrics_dir)

    train, val, test = trainer.prepare_datasets(train_df=train, val_df=val, test_df=test)
    trainer.train(train, val)
    threshold = trainer.test(val)
    trainer.test(test, threshold = threshold)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

