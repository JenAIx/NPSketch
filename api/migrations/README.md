# Database Migrations

This directory contains database migration scripts for NPSketch.

## Available Migrations

### add_image_hash.py

Adds duplicate detection capability by adding an `image_hash` column to the `uploaded_images` table.

**What it does:**
- Adds `image_hash` column (SHA256 hash of processed image)
- Creates an index on `image_hash` for fast lookups
- Calculates hashes for all existing images
- Reports potential duplicates

**How to run:**

```bash
cd api/migrations
python add_image_hash.py
```

Or with a custom database URL:

```bash
python add_image_hash.py --database-url sqlite:///path/to/your/database.db
```

**Note:** This migration is idempotent - it's safe to run multiple times.

