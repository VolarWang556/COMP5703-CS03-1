import os
import pandas as pd

# Set the dataset folder path
dataset_path = "dataset"

# Initialize an empty list to store results
results = []

# Loop through files in the dataset directory
for filename in os.listdir(dataset_path):
    # Process all model files
    if filename.endswith("_model.csv"):
        file_path = os.path.join(dataset_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Determine prefix (L or R)
        if filename.startswith("L"):
            column_name = "LFootProgressAngles_Z"
        elif filename.startswith("R"):
            column_name = "RFootProgressAngles_Z"
        else:
            continue  # Skip files with unknown leg prefix
        
        # Check if the target column exists
        if column_name in df.columns:
            # Get the total number of rows
            total_rows = len(df)
            
            # Calculate indices for 15% and 50% of the data
            start_index = int(0.15 * total_rows)
            end_index = int(0.50 * total_rows)
            
            # Extract the data between 15% and 50% of the stance phase
            data_range = df[column_name].iloc[start_index:end_index]
            
            # Compute the mean of the selected data
            mean_value = data_range.mean()
            
            # Save result
            results.append({"Filename": filename, "FPA": round(mean_value, 4)})
            
            # Optional: print the result
            print(f"File: {filename}, Column: {column_name}, Mean (15%-50%): {mean_value:.4f}")
        else:
            print(f"Warning: Column {column_name} not found in {filename}. Skipping...")


# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("FPA_values.csv", index=False)

print("\n✅ FPA results saved to 'FPA_values.csv'")