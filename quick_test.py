
"""
Quick test - 5 images to verify setup
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env")

# Read the pipeline
with open('medical_report_pipeline_kafka.py', 'r') as f:
    content = f.read()

# Modify configuration for quick test
content = content.replace('TEST_IMAGE_SIZES = [20, 400, 800, 1600]', 'TEST_IMAGE_SIZES = [5]')
content = content.replace('USE_KAFKA = True', 'USE_KAFKA = False')
content = content.replace('USE_S3_SOURCE = True', 'USE_S3_SOURCE = False')
content = content.replace('USE_S3_RESULTS = True', 'USE_S3_RESULTS = False')

# Save modified version
with open('quick_test_pipeline.py', 'w') as f:
    f.write(content)

print("✅ Created quick_test_pipeline.py with 5 images")
print("="*60)
print("QUICK TEST - Processing 5 images")
print("="*60)

# Execute it
exec(compile(open('quick_test_pipeline.py').read(), 'quick_test_pipeline.py', 'exec'))
