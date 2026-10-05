#!/usr/bin/env python3
"""Compile Strata's engine and image encoder for the cards in this machine, and nothing else.

Strata's ./setup.sh does this too, but it also wants to pick and download a model from its menu,
and the OrcaRouter build is not in that menu. This calls the same build function setup uses.
Run it with the Strata venv's python:  .venv/bin/python /path/to/build_strata_engine.py /opt/strata/Strata
"""
import os, sys

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.argv = ["setup.py"]
sys.path.insert(0, root)
os.chdir(root)
import setup as S

llama = S.get_llama_cpp()                     # the llama.cpp commit Strata pins, into third_party/
cards = S.gpus()
print("cards:", [(c.get("name"), c.get("arch")) for c in cards], flush=True)
gpu = {**cards[0], "archs": sorted({int(c["arch"]) for c in cards})}
S.build_engine(gpu, "gpu", True, llama)       # "gpu" = the image encoder runs on the card too
print("BUILD DONE", flush=True)
