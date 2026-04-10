import os, pandas as pd, uuid

def process(root, csv_path):
    df = pd.read_csv(csv_path)
    df["id"] = [uuid.uuid4().hex for _ in range(len(df))] #Add a new column with unique IDs

    for _, row in df.iterrows():
        img_name = str(row["patient_id"]) + ".tiff"
        marker = row["stain"]
        new = row["id"] + ".tiff"

        for t in ["img", "mask"]:
            p = os.path.join(root, t, marker, img_name)
            if os.path.exists(p):
                os.rename(p, os.path.join(root, t, marker, new))

    df.to_csv(csv_path, index=False)


csv_path = "clinical_data.csv"
folder = "dataset"
marker_list = ["BCL2", "BCL6", "CD10", "MUM1", "MYC", "HE"]

process(folder, csv_path)