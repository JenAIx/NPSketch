# Image Data Storage - Known Issue

## Issue: `image_data` and `processed_image_data` are the Same

### Current Behavior

Both `image_data` and `processed_image_data` columns in the `uploaded_images` table contain **the same data** (the normalized 256×256 image).

### Why This Happens

The current upload workflow is:

1. **Frontend normalization** (`upload.html`)
   - User uploads an image
   - Frontend calls `/api/normalize-image`
   - Backend normalizes to 256×256 and returns it
   - Frontend displays the normalized image

2. **User adjustments** (optional)
   - User can manually rotate, scale, translate
   - These changes are applied to the canvas

3. **Analysis** (`analyzeDrawing()`)
   - Frontend converts canvas to blob (which contains the already-normalized image)
   - Sends this blob to `/api/upload`
   - Backend receives already-normalized image

4. **Backend storage**
   ```python
   uploaded_image = UploadedImage(
       filename=filename,
       image_data=image_bytes,           # Already normalized!
       processed_image_data=processed_bytes,  # Also normalized!
       ...
   )
   ```

### Root Cause

The original upload image is **lost** during the frontend normalization step. By the time the backend receives the image for analysis, it's already been processed.

## Solutions

### Option 1: Accept Current Behavior ✅ **RECOMMENDED**

**Pros:**
- No code changes needed
- Current workflow works well
- Duplicate detection still works (uses processed_image_data)
- Normalization happens once (efficient)

**Cons:**
- Can't recover original image quality
- Both fields are redundant

**Recommendation:** This is actually fine for the current use case! Since:
- The app is designed for line drawings (not photos)
- 256×256 is the target resolution for analysis
- Duplicate detection works on processed images (which we have)
- Storage is not significantly wasted (same data, but small images)

### Option 2: Store Original Before Normalization

Modify `upload.html` to send both original and normalized images:

```javascript
async function handleFileUpload(file) {
    // Keep original file
    const originalBlob = file;
    
    // Normalize for display
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/normalize-image', ...);
    const normalizedBlob = await response.blob();
    
    // Store for later
    window.originalImageBlob = originalBlob;
    window.normalizedImageBlob = normalizedBlob;
    
    // Display normalized
    uploadedImage.src = URL.createObjectURL(normalizedBlob);
}

async function analyzeDrawing() {
    // Send BOTH images
    const formData = new FormData();
    formData.append('original_image', window.originalImageBlob);
    formData.append('processed_image', window.normalizedImageBlob);
    // ...
}
```

**Pros:**
- True original preserved
- Can provide higher resolution if needed

**Cons:**
- More complex frontend code
- Increased network traffic
- Larger database storage

### Option 3: Backend-Only Upload (No Frontend Normalization)

Remove frontend normalization and do everything in backend:

```javascript
async function handleFileUpload(file) {
    // Upload original directly
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    
    // Backend returns already-analyzed results
    const data = await response.json();
    showResults(data);
}
```

**Pros:**
- Simpler frontend
- Backend controls all processing
- Original and processed properly separated

**Cons:**
- Breaks current workflow (no manual adjustments before analysis)
- Less interactive UX
- Requires significant refactoring

## Current Status

**No action needed.** The current behavior is acceptable for the application's use case.

If you want to preserve true originals in the future, implement **Option 2** when needed.

## Verification

To check your database:

```sql
-- Compare sizes (they should be the same)
SELECT 
    id, 
    filename,
    LENGTH(image_data) as original_size,
    LENGTH(processed_image_data) as processed_size
FROM uploaded_images;

-- They will show the same size because they're identical
```

