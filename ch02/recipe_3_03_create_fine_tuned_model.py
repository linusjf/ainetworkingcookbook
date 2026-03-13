#!/usr/bin/env python
"""
Recipe303CreateFineTunedModel.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : recipe_3_03_create_fine_tuned_model
# @created     : Friday Mar 13, 2026 17:22:45 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import os
import time
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
print("Uploading training data...")

with open('bgp_training_data.jsonl', 'rb') as f:
    file_response = client.files.create(file=f, purpose='fine-tune')
print(f"File uploaded: {file_response.id}")
print("Starting fine-tuning job...")
job = client.fine_tuning.jobs.create(
training_file=file_response.id,
model="gpt-3.5-turbo",
suffix="bgp-standards"
)
print(f"Fine-tuning job created: {job.id}")
print("Waiting for completion...")
while True:
    job_status = client.fine_tuning.jobs.retrieve(job.id)
    assert job_status is not None
    if job_status.status == "succeeded":
        print(
        f"Fine-tuning completed: {job_status.fine_tuned_model}"
        )
        with open('fine_tuned_model_id.txt', 'w') as f:
            assert job_status.fine_tuned_model is not None
            f.write(job_status.fine_tuned_model)
            break
    elif job_status.status in ["failed", "cancelled"]:
        print(f"Fine-tuning {job_status.status}")
        break
    time.sleep(30)
