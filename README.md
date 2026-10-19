# MRI DeCal — Lab 4: Gradients and Spatial Encoding

Student materials for Lab 4 of the MRI DeCal. This assignment uses the Bloch simulator to collect a signal while a gradient is on, connect that signal to k-space, and reconstruct an image with a Fourier transform.

This repository does **not** include solutions.

## Requirements

- Python 3.12 or newer (`python3 --version`)
- A way to run Jupyter notebooks, such as [Cursor](https://cursor.com), [VS Code](https://code.visualstudio.com/) with the Jupyter extension, JupyterLab, or Jupyter Notebook

## Setup

1. Clone this repository and enter the folder:

```bash
git clone https://github.com/rYan-theusername/mri_decal_lab_4.git
cd mri_decal_lab_4
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the Python packages:

```bash
pip install -r requirements.txt
```

4. Open `lab_4.ipynb` and run the first cell. If it prints `All libraries are installed.`, you are ready to start.

If you prefer the Jupyter web interface:

```bash
pip install notebook
jupyter notebook lab_4.ipynb
```

## Working on the lab

- Fill in every `# TODO` in `lab_4.ipynb`.
- Short written answers go in the cyan *Your answer here.* cells.
- `bloch.py` is the spin simulator. Helper functions live in `helpers.py` and `visualizations.py`. You do not need to edit those files.
- Keep `bloch.py`, `helpers.py`, `visualizations.py`, and `top_to_bottom_cartesian.gif` in the same folder as the notebook.
- Some cells render animations. The first render of a cell can take a few seconds.

## Files

| File | What it is |
| --- | --- |
| `lab_4.ipynb` | The assignment notebook |
| `bloch.py` | Single-spin Bloch simulator used by the notebook |
| `helpers.py` | Helpers for creating objects, playing gradients, and reconstructing |
| `visualizations.py` | Plotting and animation helpers |
| `top_to_bottom_cartesian.gif` | Figure used near the end of the notebook |

If you have questions about the lab, reach out to the course staff.
