#!/usr/bin/env python3
"""
Migrate inference_results.json to MongoDB with recipe_id mapping
Usage: python migrate_inference_results.py [--file inference_results.json]
"""

import json
import os
import argparse
from pymongo import MongoClient
from bson import ObjectId, json_util
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")


def migrate_inference_results(json_file: str, drop_existing: bool = False):
    """Migrate inference_results with recipe_id mapping"""

    # Connect to MongoDB
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # Build recipe_name -> recipe_id mapping
    print("Building recipe mapping...")
    recipe_map = {}
    for recipe in db.recipes.find({}, {"_id": 1, "name": 1}):
        recipe_map[recipe["name"]] = str(recipe["_id"])
    print(f"  Found {len(recipe_map)} recipes")

    # Read JSON file (line by line for JSONL format)
    print(f"Reading {json_file}...")
    records = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line, object_hook=json_util.object_hook))
    print(f"  Found {len(records)} records")

    # Drop existing if requested
    if drop_existing:
        db.inference_results.drop()
        print("  Dropped existing inference_results collection")

    # Process and insert records
    imported = 0
    skipped = 0
    not_found = set()

    for record in records:
        recipe_name = record.get("recipe_name")
        old_recipe_id = record.get("recipe_id")

        # Get new recipe_id from mapping
        new_recipe_id = recipe_map.get(recipe_name)

        if not new_recipe_id:
            not_found.add(recipe_name)
            skipped += 1
            continue

        # Update recipe_id
        record["recipe_id"] = new_recipe_id

        # Update image_path if contains old recipe_id
        for cam in record.get("camera_results", []):
            for frame in cam.get("frames", []):
                if frame.get("image_path") and old_recipe_id:
                    frame["image_path"] = frame["image_path"].replace(old_recipe_id, new_recipe_id)

        # Generate new _id
        record["_id"] = ObjectId()

        # Insert
        db.inference_results.insert_one(record)
        imported += 1

    client.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Imported: {imported}")
    print(f"Skipped:  {skipped}")
    if not_found:
        print(f"Recipes not found: {not_found}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate inference_results with recipe mapping")
    parser.add_argument("--file", "-f", default="inference_results.json", help="JSON file path")
    parser.add_argument("--drop", "-d", action="store_true", help="Drop existing collection")

    args = parser.parse_args()
    migrate_inference_results(args.file, args.drop)
