import os
import pandas as pd
import numpy as np

# Set the dataset folder path
dataset_path = "dataset"

# Initialize an empty list to store results
results = []

# Function to calculate angle between two 2D vectors
def calculate_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

# Process all trajectory files
for filename in os.listdir(dataset_path):
    if filename.endswith("_trajectory.csv"):
        file_path = os.path.join(dataset_path, filename)
        
        # Determine prefix (L or R)
        if filename.startswith("L"):
            THI_X, THI_Z = "LTHI_X", "LTHI_Z"
            KNE_X, KNE_Z = "LKNE_X", "LKNE_Z"
            ANK_X, ANK_Z = "LANK_X", "LANK_Z"
        elif filename.startswith("R"):
            THI_X, THI_Z = "RTHI_X", "RTHI_Z"
            KNE_X, KNE_Z = "RKNE_X", "RKNE_Z"
            ANK_X, ANK_Z = "RANK_X", "RANK_Z"
        else:
            continue  # skip unknown files
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if all columns exist
        required_cols = [THI_X, THI_Z, KNE_X, KNE_Z, ANK_X, ANK_Z]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns in {filename}, skipping.")
            continue
        
        # Calculate HKA angles for each frame
        hka_angles = []
        for i in range(len(df)):
            # Get 2D points in X-Z plane
            hip = np.array([df.loc[i, THI_X], df.loc[i, THI_Z]])
            knee = np.array([df.loc[i, KNE_X], df.loc[i, KNE_Z]])
            ankle = np.array([df.loc[i, ANK_X], df.loc[i, ANK_Z]])

            # Vectors: knee→hip and knee→ankle
            vec_thigh = hip - knee
            vec_shank = ankle - knee

            # Compute angle between two vectors
            angle = calculate_angle(vec_thigh, vec_shank)
            hka_angles.append(angle)

        # Calculate deviation: max - initial
        if len(hka_angles) > 0:
            deviation = max(hka_angles) - hka_angles[0]
        else:
            deviation = np.nan

        # Save result
        results.append({"Filename": filename, "HKA_angle_deviation": round(deviation, 4)})
        print(f"{filename}: HKA angle deviation = {deviation:.4f}")

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("HKA_angle_deviation.csv", index=False)

print("\n✅ HKA angle deviation results saved to 'HKA_angle_deviation.csv'")
