"""Execute C0 -> C1 -> G1 -> G1b -> G2a -> G2b -> P1 -> P2 -> G3 -> G3b -> P4 -> D1 -> G4 against a fake Drive.

    python tools/colab_smoke/make_fake_drive.py runs/fake_drive
    python tools/colab_smoke/run_smoke.py runs/fake_drive [NOTEBOOK ...]

Fails on the first cell error; executed copies go to <root>/executed/. About 15-25 minutes on a CPU.
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def run(root, names=("C0_transfer_from_kaggle", "C1_colab_setup_check", "G1_transport_gate", "G1b_transport_diagnostics", "G2a_pool_and_streams", "G2b_textbank_and_four_ood", "P1_generation_probe", "P2_prior_probe", "G3_final_eval", "G3b_final_eval_vitl14", "P4_tins_probe", "D1_vitl14_check", "G4_reneg_on_tins")):
    root = os.path.abspath(root)
    drive = os.path.join(root, "MyDrive", "ReNeg")
    os.environ["RENEG_DRIVE_ROOT"] = drive
    os.environ["RENEG_OODLAB_HOME"] = os.path.join(root, "oodlab_home")
    os.environ["RENEG_FAKE_GEN"] = "1"
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        os.environ.pop(k, None)
    outdir = os.path.join(root, "executed")
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        if name.startswith("P2"):                                   # P2 needs a ViT-L/14 cache: reuse the fake one
            import shutil
            cache = os.path.join(drive, "oodlab_working", "cache")
            if os.path.isdir(os.path.join(cache, "vit-b-16")) and not os.path.isdir(os.path.join(cache, "vit-l-14")):
                shutil.copytree(os.path.join(cache, "vit-b-16"), os.path.join(cache, "vit-l-14"))
        src = (os.path.join(REPO, "notebooks", "1_setup_colab", f"{name}.ipynb") if name.startswith("C0")
               else os.path.join(drive, "notebooks", f"{name}.ipynb"))
        nb = nbformat.read(src, as_version=4)
        t0 = time.time()
        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": root}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, os.path.join(outdir, f"{name}.ipynb"))
        print(f"{name}: OK in {time.time() - t0:.0f}s")
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for o in cell.get("outputs", []):
                txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", ""))
                if txt:
                    print("   |", "\n   | ".join(str(txt).strip().splitlines()[-12:]))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_drive", tuple(sys.argv[2:]) or ("C0_transfer_from_kaggle", "C1_colab_setup_check", "G1_transport_gate", "G1b_transport_diagnostics", "G2a_pool_and_streams", "G2b_textbank_and_four_ood", "P1_generation_probe", "P2_prior_probe", "G3_final_eval", "G3b_final_eval_vitl14", "P4_tins_probe", "D1_vitl14_check", "G4_reneg_on_tins"))
