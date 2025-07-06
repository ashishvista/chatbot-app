#!/usr/bin/env python3
"""
Simple debugging test for path issues
"""

import os
import sys

print("🔍 DEBUGGING PATH ISSUES")
print("=" * 40)
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print()

# Check various path combinations
paths_to_check = [
    "data/documents",
    "./data/documents", 
    "src/../data/documents",
    "/Users/ashish_kumar/chat-bot/chatbot-app/data/documents"
]

for path in paths_to_check:
    exists = os.path.exists(path)
    if exists:
        file_count = len([f for f in os.listdir(path) if f.endswith(('.txt', '.md'))])
        print(f"✅ {path} - EXISTS ({file_count} files)")
    else:
        print(f"❌ {path} - NOT FOUND")

print()
print("💡 Environment variables:")
if 'DOCUMENTS_PATH' in os.environ:
    print(f"DOCUMENTS_PATH = {os.environ['DOCUMENTS_PATH']}")
else:
    print("DOCUMENTS_PATH not set")

# Test from src directory context
print()
print("🧪 Testing from different working directories:")

# Save current directory
original_cwd = os.getcwd()

# Test from src directory
src_path = os.path.join(original_cwd, 'src')
if os.path.exists(src_path):
    os.chdir(src_path)
    print(f"From src/: Current dir = {os.getcwd()}")
    
    # Check relative paths from src
    test_paths = ["../data/documents", "./data/documents", "data/documents"]
    for path in test_paths:
        exists = os.path.exists(path)
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  {path}: {status}")
    
    # Return to original directory
    os.chdir(original_cwd)

print(f"\nReturned to: {os.getcwd()}")
