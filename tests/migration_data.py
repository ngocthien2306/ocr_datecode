"""
Import/Migrate MongoDB data from exported JSON files
Usage: python migration_data.py [--input-dir ./exports] [--drop-existing]
"""

import json
import os
import argparse
import glob
from datetime import datetime
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection (target)
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")

# Collections to import
COLLECTIONS = ["recipes", "users", "cameras"]


def import_collection(db, collection_name: str, filepath: str, drop_existing: bool = False) -> dict:
    """Import a single collection from JSON file"""

    collection = db[collection_name]

    # Read JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f, object_hook=json_util.object_hook)

    if not documents:
        print(f"  [!] No documents found in {filepath}, skipping...")
        return {"collection": collection_name, "imported": 0, "skipped": 0}

    # Drop existing collection if requested
    if drop_existing:
        existing_count = collection.count_documents({})
        if existing_count > 0:
            collection.drop()
            print(f"  [!] Dropped existing '{collection_name}' ({existing_count} documents)")

    # Import documents
    imported = 0
    skipped = 0
    errors = []

    for doc in documents:
        try:
            # Check if document already exists (by _id)
            if not drop_existing and "_id" in doc:
                existing = collection.find_one({"_id": doc["_id"]})
                if existing:
                    skipped += 1
                    continue

            # Insert document
            collection.insert_one(doc)
            imported += 1
        except Exception as e:
            errors.append({"doc_id": str(doc.get("_id", "unknown")), "error": str(e)})

    print(f"  [+] Imported {imported} documents to '{collection_name}' (skipped: {skipped})")

    if errors:
        print(f"  [!] Errors: {len(errors)}")
        for err in errors[:3]:  # Show first 3 errors
            print(f"      - {err['doc_id']}: {err['error']}")

    return {
        "collection": collection_name,
        "imported": imported,
        "skipped": skipped,
        "errors": len(errors)
    }


def find_latest_export_file(input_dir: str, collection_name: str) -> str:
    """Find the latest export file for a collection"""
    pattern = os.path.join(input_dir, f"{collection_name}_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def import_all(input_dir: str = "./exports", drop_existing: bool = False, use_manifest: bool = True):
    """Import all collections from JSON files"""

    print("=" * 60)
    print("MongoDB Data Import/Migration Tool")
    print("=" * 60)
    print(f"MongoDB URL: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Input directory: {input_dir}")
    print(f"Drop existing: {drop_existing}")
    print("=" * 60)

    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        return

    # Connect to MongoDB
    try:
        client = MongoClient(MONGODB_URL)
        db = client[DATABASE_NAME]

        # Test connection
        client.admin.command('ping')
        print("\n[OK] Connected to MongoDB successfully!\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to MongoDB: {e}")
        return

    # Try to read manifest first
    manifest_path = os.path.join(input_dir, "manifest.json")
    files_to_import = {}

    if use_manifest and os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        print(f"[OK] Found manifest from: {manifest.get('export_time', 'unknown')}")
        print(f"     Source database: {manifest.get('database', 'unknown')}\n")

        for col_info in manifest.get("collections", []):
            if col_info.get("file"):
                files_to_import[col_info["collection"]] = os.path.join(input_dir, col_info["file"])
    else:
        print("[!] No manifest found, searching for latest export files...\n")

    # Import each collection
    results = []
    for collection_name in COLLECTIONS:
        # Find file to import
        filepath = files_to_import.get(collection_name)

        if not filepath or not os.path.exists(filepath):
            # Try to find latest export file
            filepath = find_latest_export_file(input_dir, collection_name)

        if not filepath:
            print(f"  [!] No export file found for '{collection_name}', skipping...")
            results.append({
                "collection": collection_name,
                "imported": 0,
                "skipped": 0,
                "error": "file not found"
            })
            continue

        print(f"  Processing: {os.path.basename(filepath)}")
        result = import_collection(db, collection_name, filepath, drop_existing)
        results.append(result)

    # Close connection
    client.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Import Summary:")
    print("=" * 60)
    total_imported = sum(r.get("imported", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    print(f"  Total documents imported: {total_imported}")
    print(f"  Total documents skipped: {total_skipped}")
    for r in results:
        if r.get("error"):
            print(f"  - {r['collection']}: {r['error']}")
        else:
            print(f"  - {r['collection']}: {r['imported']} imported, {r['skipped']} skipped")
    print(f"\nImport completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Import/Migrate MongoDB data from JSON files")
    parser.add_argument(
        "--input-dir", "-i",
        default="./exports",
        help="Input directory containing exported JSON files (default: ./exports)"
    )
    parser.add_argument(
        "--drop-existing", "-d",
        action="store_true",
        help="Drop existing collections before importing (WARNING: destructive!)"
    )
    parser.add_argument(
        "--mongodb-url",
        default=None,
        help="MongoDB URL (overrides .env)"
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Database name (overrides .env)"
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Don't use manifest, find latest files automatically"
    )

    args = parser.parse_args()

    # Override globals if provided
    global MONGODB_URL, DATABASE_NAME
    if args.mongodb_url:
        MONGODB_URL = args.mongodb_url
    if args.database:
        DATABASE_NAME = args.database

    import_all(
        input_dir=args.input_dir,
        drop_existing=args.drop_existing,
        use_manifest=not args.no_manifest
    )


if __name__ == "__main__":
    main()
