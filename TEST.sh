#!/bin/sh

set -e

pip install -U safety
safety check -r requirements.txt
python -m pytest --disable-pytest-warnings .
