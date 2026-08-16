from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import time
import unicodedata
from contextlib import redirect_stderr, redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
