#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥