

# Whiteboard Detection and Text Identification

For quick testing of the app, visit this [link](https://sabaina-haroon-whiteboard-detection-and-text-i-inference-jiy28u.streamlitapp.com/)


# Installation Guide
<a name="TOP"></a>

### 1. Install Tesseract 
<a name="TOP"></a>
- <b> <span style="color: Green; "> Ubuntu </span> </b>
       
You can install Tesseract and its developer tools on Ubuntu by simply running:
    
    
        sudo apt install tesseract-ocr
        sudo apt install libtesseract-dev
Note for Ubuntu users: In case apt is unable to find the package try adding universe entry to the sources.list file as shown below.


    sudo vi /etc/apt/sources.list

    Copy the first line "deb http://archive.ubuntu.com/ubuntu bionic main" and paste it as shown below on the next line.
    If you are using a different release of ubuntu, then replace bionic with the respective release name.

    deb http://archive.ubuntu.com/ubuntu bionic universe

<a name="TOP"></a>
- <b> <span style="color: Green; "> Windows </span> </b>
        
Installer for Windows for Tesseract 3.05, Tesseract 4 and Tesseract 5 are available from <a href="https://github.com/UB-Mannheim/tesseract/wiki" target="https://github.com/UB-Mannheim/tesseract/wiki">Tesseract at UB Mannheim</a> . These include the training tools. Both 32-bit and 64-bit installers are available.
    
Once the installer is downloaded and installed, add the directory where the tesseract-OCR binaries are located to the Path variables. Default directory for tesseract is 

    C:\Program Files\Tesseract-OCR.

### 2. Install Dependencies 

Clone repo and install [requirements.txt](https://github.com/sabaina-Haroon/whiteboard_detection_and_text_identification/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/sabaina-Haroon/whiteboard_detection_and_text_identification  # clone
pip install -r requirements.txt  # install
```

### 3. Run the App 
In the terminal, type command;

    streamlit run inference.py