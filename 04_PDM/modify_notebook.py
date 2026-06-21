import json

with open("notebooks/run_pdm_colab.ipynb", "r") as f:
    nb = json.load(f)

# Modify cell 6 (index 6) to only unpack code
cell_6 = nb["cells"][6]
cell_6["source"] = [
    "import os, shutil, time\n",
    "\n",
    "DRIVE_DIR = '/content/drive/MyDrive/pdm'   # <-- folder holding the code zip\n",
    "CODE_ZIP      = f'{DRIVE_DIR}/pdm_code.zip'\n",
    "assert os.path.exists(CODE_ZIP), f'missing {CODE_ZIP} — upload it to {DRIVE_DIR}'\n",
    "\n",
    "t0 = time.time()    \n",
    "!cp '{CODE_ZIP}' /content/\n",
    "!mkdir -p /content/pdm_code && unzip -q -o /content/pdm_code.zip -d /content/pdm_code\n",
    "print(f'code unpacked in {time.time()-t0:.0f}s')\n",
    "!ls /content/pdm_code\n"
]

# Insert a new markdown cell and code cell for preprocessing
preprocess_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3.5. Run Preprocessing on Colab (since upload is too slow)\n",
        "Point `PDM_DATA_ROOT` to where the raw BraTS data is located on your Google Drive.\n",
        "This reads the NIfTI volumes from Drive and writes the `.npy` slices to the fast local `/content/processed/` disk."
    ]
}

preprocess_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /content/pdm_code\n",
        "# Change this path to where your raw BraTS dataset is in Drive\n",
        "%env PDM_DATA_ROOT=/content/drive/MyDrive/BraTS-PEDs-v1/Training\n",
        "%env PDM_PROCESSED_ROOT=/content/processed\n",
        "\n",
        "!python scripts/00_preprocess.py --splits splits\n"
    ]
}

nb["cells"].insert(7, preprocess_md)
nb["cells"].insert(8, preprocess_code)

with open("notebooks/run_pdm_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

