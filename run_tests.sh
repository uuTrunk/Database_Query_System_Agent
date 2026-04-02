#!/bin/bash
echo "Running Agent Backend tests..."
conda run -n dqs python main.py test api.tests
