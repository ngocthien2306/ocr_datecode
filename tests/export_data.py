"""
Export MongoDB collections (recipes, users, cameras) to JSON files
Usage: python export_data.py [--output-dir ./exports]
"""

import json
import os
import argparse
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId, json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")

# Collections to export
COLLECTIONS = ["recipes", "users", "cameras"]


class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB types"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return {"$oid": str(obj)}
        if isinstance(obj, datetime):
            return {"$date": obj.isoformat()}
        return super().default(obj)


def export_collection(db, collection_name: str, output_dir: str) -> dict:
    """Export a single collection to JSON file"""
    collection = db[collection_name]

    # Get all documents
    documents = list(collection.find())
    count = len(documents)

    if count == 0:
        print(f"  [!] Collection '{collection_name}' is empty, skipping...")
        return {"collection": collection_name, "count": 0, "file": None}

    # Create output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{collection_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Write to JSON file using bson json_util for proper MongoDB type handling
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(documents, f, default=json_util.default, ensure_ascii=False, indent=2)

    print(f"  [+] Exported {count} documents from '{collection_name}' -> {filename}")

    return {
        "collection": collection_name,
        "count": count,
        "file": filename
    }


def export_all(output_dir: str = "./exports"):
    """Export all collections to JSON files"""

    print("=" * 60)
    print("MongoDB Data Export Tool")
    print("=" * 60)
    print(f"MongoDB URL: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Collections: {', '.join(COLLECTIONS)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

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

    # Export each collection
    results = []
    for collection_name in COLLECTIONS:
        result = export_collection(db, collection_name, output_dir)
        results.append(result)

    # Create manifest file
    manifest = {
        "export_time": datetime.now().isoformat(),
        "mongodb_url": MONGODB_URL,
        "database": DATABASE_NAME,
        "collections": results
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Close connection
    client.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Export Summary:")
    print("=" * 60)
    total_docs = sum(r["count"] for r in results)
    print(f"  Total documents exported: {total_docs}")
    for r in results:
        status = f"{r['count']} docs" if r['count'] > 0 else "empty"
        print(f"  - {r['collection']}: {status}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"Export completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export MongoDB collections to JSON")
    parser.add_argument(
        "--output-dir", "-o",
        default="./exports",
        help="Output directory for exported files (default: ./exports)"
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

    args = parser.parse_args()

    # Override globals if provided
    global MONGODB_URL, DATABASE_NAME
    if args.mongodb_url:
        MONGODB_URL = args.mongodb_url
    if args.database:
        DATABASE_NAME = args.database

    export_all(args.output_dir)


if __name__ == "__main__":
    main()
