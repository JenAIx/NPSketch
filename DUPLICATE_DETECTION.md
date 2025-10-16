# Duplicate Image Detection Feature

## Overview

NPSketch now detects duplicate images using content-based hashing (SHA256) to prevent storing the same image multiple times in the database.

## How It Works

1. **Hash Calculation**: When an image is uploaded and normalized, a SHA256 hash is calculated from the processed image data.

2. **Duplicate Check**: Before storing a new image, the system checks if an image with the same hash already exists.

3. **User Warning**: If a duplicate is detected, the user sees a warning banner with information about the existing image.

4. **Existing Results**: When analyzing a duplicate, the system returns the existing evaluation results instead of creating new entries.

## User Experience

### Upload Flow

1. User uploads an image via `upload.html`
2. Image is normalized to 256×256
3. System calculates hash and checks for duplicates
4. If duplicate:
   - **Orange warning banner** appears
   - Shows when the image was originally uploaded
   - Shows who uploaded it
   - Informs user that analysis will use existing data
   - User can still proceed with analysis

### Warning Banner

```
⚠️ Duplicate Image Detected

This image already exists in the database:
Original upload: October 16, 2025, 3:45:23 PM
Uploaded by: John Doe

ℹ️ You can still analyze this image, but it will not be 
stored again to avoid duplicates. The existing evaluation 
results will be returned.
```

## Technical Implementation

### Database Changes

**New Column**: `uploaded_images.image_hash`
- Type: VARCHAR(64)
- Nullable: Yes (for backward compatibility)
- Indexed: Yes (for fast lookups)

### API Changes

**New Endpoint**: `POST /api/check-duplicate`
- Accepts: Image file
- Returns: Duplicate status and existing image info

**Modified Service**: `EvaluationService.process_upload()`
- Calculates hash before storing
- Checks for existing images
- Returns existing evaluation if duplicate found

### Frontend Changes

**Modified**: `webapp/upload.html`
- Calls `/api/check-duplicate` after normalization
- Shows warning banner if duplicate detected
- Tracks duplicate state throughout session
- Clears warning when starting new upload

## Database Migration

For existing databases, run the migration script:

```bash
cd api/migrations
python add_image_hash.py
```

This will:
- Add the `image_hash` column
- Create an index for performance
- Calculate hashes for all existing images
- Report any existing duplicates

## Benefits

✅ **Prevents Duplicate Storage**: Saves database space  
✅ **User Awareness**: Users know when they're re-uploading  
✅ **Data Consistency**: Uses existing evaluations for duplicates  
✅ **Fast Detection**: Indexed hash lookups are very quick  
✅ **Content-Based**: Detects duplicates even if filename differs  

## Future Enhancements

Potential improvements:
- Show similar (near-duplicate) images
- Allow users to view existing evaluation when duplicate detected
- Duplicate management page (view/delete duplicates)
- Configurable duplicate handling (allow/prevent/merge)

