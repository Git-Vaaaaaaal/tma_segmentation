import os, random, shutil

def split_dataset(root, out, split=0.9):
    markers = ["BCL2","BCL6","CD10","HE","MUM1","MYC"]
    
    for t in ["train","test"]:
        for d in ["img","mask"]:
            os.makedirs(os.path.join(out, t, d), exist_ok=True)

    for m in markers:
        img_dir = os.path.join(root, "img", m)
        mask_dir = os.path.join(root, "mask", m)

        files = [f for f in os.listdir(img_dir) if f.endswith(".tiff")]
        random.shuffle(files)
        cut = int(len(files) * split)

        splits = {"train": files[:cut], "test": files[cut:]}

        for s in ["train","test"]:
            for f in splits[s]:
                shutil.copy(os.path.join(img_dir, f),
                            os.path.join(out, s, "img", f))
                
                mpath = os.path.join(mask_dir, f)
                if os.path.exists(mpath):
                    shutil.copy(mpath,
                                os.path.join(out, s, "mask", f))


split_dataset("dataset_raw", "dataset_unet")
