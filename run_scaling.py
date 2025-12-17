"""
Full scaling test: 20, 50, 100, 200, 500, 1000 images
"""
from dotenv import load_dotenv
import os

load_dotenv(".env")

with open('medical_report_pipeline_kafka.py', 'r') as f:
    content = f.read()

# Modify for scaling test
content = content.replace('TEST_IMAGE_SIZES = [20, 400, 800, 1600]', 
                         'TEST_IMAGE_SIZES = [20, 50, 100, 200, 500, 1000]')
content = content.replace('USE_KAFKA = True', 'USE_KAFKA = False')
content = content.replace('USE_S3_SOURCE = True', 'USE_S3_SOURCE = False')
content = content.replace('USE_S3_RESULTS = True', 'USE_S3_RESULTS = False')

with open('scaling_test_pipeline.py', 'w') as f:
    f.write(content)

print("="*60)
print("SCALING TEST")
print("="*60)
print("Test sizes: [20, 50, 100, 200, 500, 1000] images")
print("Estimated time: 2-3 hours")
print("="*60)

exec(compile(open('scaling_test_pipeline.py').read(), 'scaling_test_pipeline.py', 'exec'))