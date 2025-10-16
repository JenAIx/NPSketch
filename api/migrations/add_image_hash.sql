-- Add image_hash column to uploaded_images table
-- This migration adds duplicate detection capability

-- Add the column (nullable for backward compatibility)
ALTER TABLE uploaded_images ADD COLUMN image_hash VARCHAR(64);

-- Create an index for fast lookups
CREATE INDEX IF NOT EXISTS ix_uploaded_images_image_hash 
ON uploaded_images (image_hash);

-- Note: To calculate hashes for existing images, you need to run
-- the Python script inside the Docker container or manually update them

