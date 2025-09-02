import os
import torch
import oskp_rl, oskp_rl_fixed, oskp_rl_up, oskp_rl_up_with_buffer, oskp_rl_up_with_buffer_with_mask
import pandas as pd
import analysis

# List of instance files
cut_files = ['cut_1.pt', 'cut_2.pt', 'rs.pt']
models = [oskp_rl, oskp_rl_fixed, oskp_rl_up, oskp_rl_up_with_buffer, oskp_rl_up_with_buffer_with_mask]
models_str= ['oskp_rl', 'oskp_rl_fixed', 'oskp_rl_up', 'oskp_rl_up_with_buffer', 'oskp_rl_up_with_buffer_with_mask']
#models = ['oskp_rl_up_with_buffer_with_mask']

count=0
for model in models:
    for file in cut_files:
        
        # Load the instance (checkpoint) from the approachesO3DKP folder.
        file_path = os.path.join("approachesO3DKP", file)
        instance = torch.load(file_path, weights_only=True)

        print("Running model ",models_str[count]," with instances ",file)
        
        print(f"Processing {file}")
        print("Total batches:", len(instance))
        print("Boxes in first batch:", len(instance[0]))
        
        # Create an output folder named after the file (without the extension)
        folder_name = os.path.splitext(file)[0]
        output_dir = os.path.join(os.getcwd(), folder_name+"_"+models_str[count].removeprefix('oskp_rl_'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Run training for this instance.
        # IMPORTANT: Modify your oskp_rl.train function so that it accepts an argument (e.g. output_dir)
        # and saves all visualization files (pallet images and trend graph) in that folder.
        # Also, have the training function return a pandas DataFrame with the final metrics.
        final_metrics = model.train(instance, output_dir=output_dir, verbose=True, episode_to_show=100)
        
        # Save final metrics to CSV in the output folder.
        csv_path = os.path.join(output_dir, "final_metrics.csv")
        final_metrics.to_csv(csv_path, index=False)
        
        print(f"Training for {file} completed. Visualizations and CSV saved in {output_dir}\n")
    count+=1


root = "/Users/sebastian.herrera/packing_problem_with_online_rl/"   # the directory that contains cut_1, cut_1_fixed, rs, rs_up, etc.
summary_df = analysis.summarize_variants(root, save_csv=os.path.join(root, "summary_variants.csv"))
print(summary_df.to_string(index=False))
