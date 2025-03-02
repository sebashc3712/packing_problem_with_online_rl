import os
import torch
import oskp_rl
import pandas as pd

# List of instance files
cut_files = ['cut_1.pt', 'cut_2.pt', 'rs.pt']

for file in cut_files:
    # Load the instance (checkpoint) from the approachesO3DKP folder.
    file_path = os.path.join("approachesO3DKP", file)
    instance = torch.load(file_path, weights_only=True)
    
    print(f"Processing {file}")
    print("Total batches:", len(instance))
    print("Boxes in first batch:", len(instance[0]))
    
    # Create an output folder named after the file (without the extension)
    folder_name = os.path.splitext(file)[0]
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training for this instance.
    # IMPORTANT: Modify your oskp_rl.train function so that it accepts an argument (e.g. output_dir)
    # and saves all visualization files (pallet images and trend graph) in that folder.
    # Also, have the training function return a pandas DataFrame with the final metrics.
    final_metrics = oskp_rl.train(instance, output_dir=output_dir)
    
    # Save final metrics to CSV in the output folder.
    csv_path = os.path.join(output_dir, "final_metrics.csv")
    final_metrics.to_csv(csv_path, index=False)
    
    print(f"Training for {file} completed. Visualizations and CSV saved in {output_dir}\n")
