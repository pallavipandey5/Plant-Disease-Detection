import os
import random

for d in random.sample(dataset_dicts, 3):
    #print(d)
    if not os.path.exists(d["file_name"]):
        print(f"File not found: {d['file_name']}")
        continue

    img = cv2.imread(d["file_name"])
    if img is None:
        print(f"Unable to read image: {d['file_name']}")
        continue
    
    v = Visualizer(img[:, :, ::-1], metadata=leaf_metadata, scale=0.5)
    v = v.draw_dataset_dict(d)
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()