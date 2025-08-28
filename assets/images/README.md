# Images Directory

This directory contains image assets for the research portfolio website.

## Required Images

### Profile Image
- **File**: `profile.jpg`
- **Recommended size**: 400x400px or larger (square aspect ratio)
- **Format**: JPG, PNG, or WebP
- **Description**: Your professional headshot or profile photo

### Favicon
- **File**: `favicon.ico`
- **Size**: 32x32px (standard favicon size)
- **Format**: ICO format
- **Description**: Small icon displayed in browser tabs

## Optional Images

### Research Paper Thumbnails
- **Directory**: `papers/thumbnails/`
- **Naming**: Match your paper slug (e.g., `deep-learning-nlp.jpg`)
- **Size**: 300x200px (3:2 aspect ratio)
- **Format**: JPG or PNG

### Background Images
- **Directory**: `backgrounds/`
- **Usage**: Hero section backgrounds, page headers
- **Size**: 1920x1080px or larger
- **Format**: JPG or WebP for better compression

## Image Optimization Tips

1. **Compress images** before uploading to reduce file size
2. **Use WebP format** when possible for better compression
3. **Provide alt text** for accessibility
4. **Use responsive images** with multiple sizes when needed

## Tools for Image Optimization

- [TinyPNG](https://tinypng.com/) - Online image compression
- [Squoosh](https://squoosh.app/) - Google's image optimization tool
- [ImageOptim](https://imageoptim.com/) - Mac app for image optimization

## Adding Images

### Profile Image
1. Add your profile photo as `profile.jpg` in this directory
2. Update the path in `_config.yml` if using a different name:
   ```yaml
   researcher:
     profile_image: "/assets/images/your-photo.jpg"
   ```

### Paper PDFs
1. Create a `papers` subdirectory
2. Add PDF files with descriptive names
3. Reference them in your research markdown files:
   ```yaml
   pdf_url: "/assets/papers/your-paper-2024.pdf"
   ```

### Favicon
1. Create a favicon using online tools like [Favicon.io](https://favicon.io/)
2. Save as `favicon.ico` in this directory
3. The template will automatically reference it

## Image Guidelines

- **Profile photos**: Professional, high-quality, good lighting
- **Academic context**: Appropriate for professional/academic setting
- **File sizes**: Keep under 1MB for web performance
- **Copyright**: Ensure you have rights to use all images

## Placeholder Images

If you don't have images ready, you can use placeholder services:

- Profile: `https://via.placeholder.com/400x400/1e3a8a/ffffff?text=Your+Name`
- Papers: `https://via.placeholder.com/300x200/3b82f6/ffffff?text=Paper+Title`

Replace these with actual images when available.