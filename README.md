# UniMR

# UniMR User Guide

This software is designed to segment, enhance, and match molecular images using image processing and machine learning techniques. Users can upload molecular images, select target molecules, and obtain classification results.

## System Requirements

To use our software, you need to use the following systems and install the corresponding environment.

- Operating System: Windows
- Python Version: Python 3.9 or higher

### Dependencies

- numpy==1.21.6
- Pillow==9.5.0
- scikit-learn==1.0.2
- scipy==1.7.3
- torch==1.13.1
- torchaudio==0.13.1
- torchvision==0.14.1
- tqdm==4.66.4
- clip==1.0
- realesrgan==0.3.0

Installation command:

```bash
pip install -r requirements.txt
```
## Usage

This section will introduce how to use the UniMR software.

1. **Launch the UniMR Software**

   Upon running the software, the Start Window will appear. Ensure that your Python environment is properly configured and that all dependent libraries are installed.

2. **Load Molecular Images**

   In the Start Window, click the "Select Image Path" button to select the file path containing the target molecular image. The software will load the image and display it in the Image Display Window.

3. **Select Target Molecules**

   In the Image Display Window, click the "Select Target Molecule" button, then click on the molecule you want to use as a sample in the image. The software will prompt you to enter the target molecule's name and color. After entering the details, click the "Submit" button, then the target molecule will be marked and saved. Multiple targets can be selected one by one.

4. **Classify Molecules**

   Click the "Confirm" button in the Start Window, and the software will vectorize all molecules and match them with the target molecules. If the match result is above the set threshold, the molecule will be classified as the corresponding target molecule; otherwise, it will be classified as "other".

5. **View Results**

   After classification, the software will pop up a Classification Results Window showing the quantity of each type of molecule.
