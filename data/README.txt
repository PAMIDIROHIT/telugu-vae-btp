Steps to produce and prepare the dataset (Ubuntu):

1. Ensure Pothana2000.ttf is installed at ~/.local/share/fonts/Pothana2000.ttf
   If not installed, copy font and refresh fonts:
     mkdir -p ~/.local/share/fonts
     cp Pothana2000.ttf ~/.local/share/fonts/
     fc-cache -fv

2. Ensure ocropus-linegen is installed and on PATH.

3. Place the following files in the same folder:
   - telugu_labels.txt
   - telugu_lines.txt
   - generate_ocropus_dataset.sh
   - postprocess_organize_and_resize.py
   - split_train_test.sh
   - split_train_test.py
   - verify_dataset.sh

4. Make scripts executable:
   chmod +x generate_ocropus_dataset.sh split_train_test.sh verify_dataset.sh
   chmod +x postprocess_organize_and_resize.py split_train_test.py

5. Generate images using OCROPUS:
   bash generate_ocropus_dataset.sh

6. Post-process images into class folders and resize to 32x32:
   python3 postprocess_organize_and_resize.py

7. Split into train/test (75% train, 25% test):
   bash split_train_test.sh

8. Verify counts:
   bash verify_dataset.sh

Notes:
- postprocess_organize_and_resize.py uses telugu_labels.txt to map characters to class IDs.
- If any characters in ocropus .gt.txt files do not match the labels mapping, they will be skipped and a warning printed.
- final_dataset/ will contain train/ and test/ subfolders with class_xxx folders inside.
