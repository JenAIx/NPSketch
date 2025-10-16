# Database Migration Guide

## Running the Image Hash Migration

The `image_hash` column is needed for duplicate detection. Here's how to add it to your database.

### Method 1: Using the Admin Web Interface ⭐ **EASIEST**

1. **Open your browser** and navigate to:
   ```
   http://localhost:8000/admin.html
   ```

2. **Click the "🚀 Run Migration" button**

3. **View results:**
   - Column added status
   - Number of images updated
   - Duplicate detection results
   - Statistics

That's it! The migration runs automatically via the API.

### Method 2: Direct API Call

Use curl or any HTTP client:

```bash
curl -X POST http://localhost:8000/api/admin/migrate-add-image-hash
```

### Method 3: SQL Script (If API is Down)

If you need to add the column directly to the database:

1. **Stop the API/Docker container:**
   ```bash
   docker-compose down
   ```

2. **Run the SQL migration:**
   ```bash
   cd /home/ste/MyProjects/npsketch
   sqlite3 data/npsketch.db < api/migrations/add_image_hash.sql
   ```

3. **Restart the API:**
   ```bash
   docker-compose up -d
   ```

4. **Calculate hashes for existing images:**
   - Open `http://localhost:8000/admin.html`
   - Click "Run Migration" (it will only update missing hashes)

## What the Migration Does

1. ✅ Adds `image_hash` column (VARCHAR(64)) to `uploaded_images` table
2. ✅ Creates an index on `image_hash` for fast lookups
3. ✅ Calculates SHA256 hashes for all existing images
4. ✅ Reports any duplicate images found

## Verification

After running the migration, check your database:

```bash
sqlite3 data/npsketch.db "PRAGMA table_info(uploaded_images);"
```

You should see `image_hash` in the output.

## Troubleshooting

### "Column already exists"
This is fine! The migration is safe to run multiple times. It will only update images that don't have hashes yet.

### "Database is locked"
The API is running. Use **Method 1** (Admin Web Interface) instead, which runs the migration through the API.

### "No module named 'sqlalchemy'"
Don't run the Python migration script directly. Use **Method 1** (Admin Web Interface) or **Method 3** (SQL Script).

## About the Duplicate Data Issue

You may notice that `image_data` and `processed_image_data` contain the same data. This is **expected** in the current workflow because:

1. The frontend normalizes images before sending to the backend
2. Both fields end up storing the normalized 256×256 image
3. This is fine for the application's use case

See `DUPLICATE_DATA_ISSUE.md` for details and potential solutions if you want to change this behavior.

