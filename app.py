from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
from scipy import ndimage

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

# Character sets (unchanged)
ASCII_STANDARD = " .:-=+*#%@$>"
ASCII_EXTENDED = " .'`^\",:;Il!i>`<`~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
ASCII_BLOCKS = " ░▒▓█"
ASCII_DETAILED = " .`-_':,;^=+/\"|)\\<>)iv%xclrs{*}I?!][1taeo7zjLunT#JCwfy325Fh9kpq4VWXKAmOg6PdQDbUG8RNM@"


def analyze_image(img):
    """
    🧠 MEMORY-OPTIMIZED IMAGE ANALYSIS
    Downscale ONLY for analysis, not for conversion!
    """
    # Create a smaller copy JUST for analysis
    width, height = img.size
    if max(width, height) > 600:
        if width > height:
            new_width = 600
            new_height = int((600 / width) * height)
        else:
            new_height = 600
            new_width = int((600 / height) * width)
        analysis_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        analysis_img = img
    
    gray = analysis_img.convert("L")
    # ✨ USE UINT8 for analysis
    pixels = np.array(gray, dtype=np.uint8)
    pixels_float = pixels.astype(np.float32)
    
    # Basic stats
    mean = float(pixels_float.mean())
    std = float(pixels_float.std())
    
    # Key percentiles
    p10, p50, p90 = np.percentile(pixels, [10, 50, 90])
    
    # Simple metrics
    dynamic_range = float((p90 - p10) / 255.0)
    texture = float(std / 255.0)
    
    # Edge detection (on smaller array)
    sobel_x = ndimage.sobel(pixels_float, axis=1)
    sobel_y = ndimage.sobel(pixels_float, axis=0)
    edges = np.hypot(sobel_x, sobel_y)
    edge_score = float(edges.mean() / 50.0)
    
    # Clean up intermediate arrays
    del sobel_x, sobel_y, edges, pixels_float
    
    bright_ratio = float((pixels > 180).sum() / pixels.size)
    
    settings = {}
    
    # Contrast
    if mean < 85:
        settings['contrast'] = 1.0
    elif dynamic_range < 0.35:
        settings['contrast'] = 2.2
    elif dynamic_range < 0.5:
        settings['contrast'] = 1.7
    elif dynamic_range > 0.75:
        settings['contrast'] = 1.2
    else:
        settings['contrast'] = 1.5
    
    # Brightness
    if bright_ratio > 0.25 or mean > 145:
        if mean > 155 or bright_ratio > 0.35:
            settings['brightness'] = -25
        else:
            settings['brightness'] = -20
    elif mean < 90:
        if mean < 70:
            settings['brightness'] = 30
        else:
            settings['brightness'] = 18
    elif mean < 105:
        settings['brightness'] = 10
    elif mean > 135:
        settings['brightness'] = -12
    else:
        settings['brightness'] = 0
    
    # Edge detection
    if edge_score < 0.2:
        settings['edge_detection'] = 0.2
    elif edge_score < 0.5:
        settings['edge_detection'] = 0.15
    elif edge_score > 1.0:
        settings['edge_detection'] = 0.0
    else:
        settings['edge_detection'] = 0.1
    
    settings['detail'] = 'low'
    settings['dithering'] = True
    settings['invert'] = bool(mean < 60)
    
    # Gamma
    if mean < 100:
        gamma = 0.85
    elif mean > 150:
        gamma = 1.15
    else:
        gamma = 1.0
    
    settings['gamma'] = float(gamma)
    
    settings['analysis'] = {
        'mean_brightness': round(mean, 2),
        'std_dev': round(std, 2),
        'dynamic_range': round(dynamic_range, 2),
        'texture': round(texture, 2),
        'edge_score': round(edge_score, 2),
        'bright_ratio': round(bright_ratio, 3),
        'p50': round(p50, 2),
        'p90': round(p90, 2),
        'gamma': round(gamma, 2)
    }
    
    return settings


def apply_edge_detection(img, strength=1.0):
    """
    ⚡ HIGH-QUALITY EDGE DETECTION with light smoothing
    Uses float32 for calculations, keeps quality high!
    """
    if strength == 0:
        return img

    # Convert to float32 (lighter than float64!)
    arr = np.array(img, dtype=np.float32)

    # 🔥 LIGHT smoothing to reduce noise WITHOUT killing detail
    # sigma=0.5 is MUCH lighter than before!
    smoothed = ndimage.gaussian_filter(arr, sigma=0.5)
    
    sobel_x = ndimage.sobel(smoothed, axis=1)
    sobel_y = ndimage.sobel(smoothed, axis=0)
    edges = np.hypot(sobel_x, sobel_y)
    
    # Clean up intermediate arrays
    del sobel_x, sobel_y, smoothed

    # Normalize
    max_val = edges.max()
    if max_val > 0:
        edges = (edges / max_val) * 255.0
        # Boost curve for POP! ✨
        edges = np.sqrt(edges / 255.0) * 255.0

    edges = edges.astype(np.uint8)

    # Blend
    if strength < 1.0:
        blended = (arr * (1 - strength) + edges * strength).astype(np.uint8)
        del edges, arr
        return Image.fromarray(blended)
    else:
        del arr
        return Image.fromarray(edges)


def apply_dithering(img):
    """
    🎨 HYBRID DITHERING - Best quality + good memory!
    Uses Pillow for small images, optimized custom for large ones
    """
    width, height = img.size
    total_pixels = width * height
    
    # For smaller images (< 1MP), use custom Floyd-Steinberg for BEST quality
    if total_pixels < 1_000_000:
        return apply_dithering_custom(img)
    else:
        # For huge images, use Pillow's fast version
        dithered = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
        return dithered.convert("L")


def apply_dithering_custom(img):
    """
    ✨ OPTIMIZED custom Floyd-Steinberg - high quality!
    Uses in-place operations to minimize memory
    """
    arr = np.array(img, dtype=np.float32)  # float32, not float64!
    height, width = arr.shape

    # Floyd-Steinberg error diffusion
    for y in range(height):
        for x in range(width):
            old_pixel = arr[y, x]
            new_pixel = 255.0 if old_pixel > 127 else 0.0
            arr[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Distribute error (with bounds checking)
            if x + 1 < width:
                arr[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    arr[y + 1, x - 1] += error * 3 / 16
                arr[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    arr[y + 1, x + 1] += error * 1 / 16

    # Clip and convert
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_image_enhancements(img, contrast=1.0, brightness=0, sharpness=1.0):
    """Apply enhancements to make ASCII look BETTER! ✨"""
    if contrast != 1.0:
        contrast = max(0.0, contrast)
        img = ImageEnhance.Contrast(img).enhance(contrast)
    
    if brightness != 0:
        brightness_factor = 1.0 + max(-1.0, min(brightness / 100.0, 10.0))
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    
    if sharpness != 1.0:
        sharpness = max(0.0, sharpness)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    return img


def get_char_set(detail_level='high'):
    """Get character set based on detail level"""
    sets = {
        'low': ASCII_STANDARD,
        'medium': ASCII_EXTENDED,
        'high': ASCII_DETAILED,
        'blocks': ASCII_BLOCKS
    }
    return sets.get(detail_level, ASCII_DETAILED)


def pixel_to_char(pixel_value, char_set, invert=False, gamma=1.0):
    """Convert pixel to character with gamma correction"""
    pixel_value = max(0, min(pixel_value, 255))
    
    if invert:
        pixel_value = 255 - pixel_value
    
    normalized = pixel_value / 255.0
    gamma_corrected = pow(normalized, gamma)
    
    char_index = int(gamma_corrected * (len(char_set) - 1))
    char_index = max(0, min(char_index, len(char_set) - 1))
    
    return char_set[char_index]


def image_to_ascii(
    img, max_width=100, contrast=1.0, brightness=0, invert=False, 
    detail='high', edge_detection=0.0, dithering=False
):
    """
    🔥 HIGH-QUALITY ASCII CONVERSION with smart memory usage!
    - NO aggressive downscaling before processing
    - Smart resizing based on target width
    - Uses float32 instead of float64
    - Minimal array copies
    """
    # Convert to grayscale
    img = img.convert("L")
    
    # Apply enhancements FIRST (on full resolution for quality!)
    img = apply_image_enhancements(img, contrast, brightness, sharpness=1.5)
    
    # Edge detection (if needed)
    if edge_detection > 0:
        img = apply_edge_detection(img, edge_detection)
    
    # Dithering (smart hybrid approach)
    if dithering:
        img = apply_dithering(img)
    
    # 🎯 NOW resize to target ASCII dimensions
    # This is the ONLY resize - from original to final!
    aspect_ratio = img.height / img.width
    new_height = max(1, int(max_width * aspect_ratio * 0.55))
    
    # High-quality resize
    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # ✨ USE UINT8 for final conversion
    pixels = np.array(img, dtype=np.uint8)
    
    # Select character set
    char_set = get_char_set(detail)
    
    # 🎯 GENERATE ASCII efficiently
    ascii_lines = []
    for row in pixels:
        line = ''.join(pixel_to_char(pixel, char_set, invert) for pixel in row)
        ascii_lines.append(line)
    
    # Clean up
    del pixels
    
    return '\n'.join(ascii_lines)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "alive",
        "message": "TXTfier backend is RUNNING! 🚀",
        "version": "2.3 - HIGH QUALITY + OPTIMIZED! 💎"
    })


@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    """Analyze image and suggest optimal settings"""
    try:
        file = request.files.get('image')   
        if not file or file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image uploaded!" if not file else "Empty filename!"
            }), 400

        try:
            with Image.open(file.stream) as img:
                if img.mode == 'RGBA':
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg.copy()
                else:
                    img = img.copy()
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }), 400

        settings = analyze_image(img)

        return jsonify({
            "success": True,
            "settings": settings,
            "message": "Analysis complete! 🎯"
        })

    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }), 500


@app.route('/convert', methods=['POST'])
def convert_image():
    """Main conversion endpoint - HIGH QUALITY + OPTIMIZED! 💎"""
    try:
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image uploaded!" if not file else "Empty filename! 😢"
            }), 400

        try:
            max_width = max(10, min(int(request.form.get('width', 100)), 1000))
            contrast = max(0.5, min(float(request.form.get('contrast', 1.0)), 3.0))
            brightness = max(-100, min(int(request.form.get('brightness', 0)), 100))
            invert = request.form.get('invert', 'false').lower() == 'true'
            detail = request.form.get('detail', 'high')
            edge_detection = max(0.0, min(float(request.form.get('edge_detection', 0.0)), 1.0))
            dithering = request.form.get('dithering', 'false').lower() == 'true'
        except ValueError as e:
            return jsonify({"success": False, "error": f"Invalid parameter: {str(e)}"}), 400

        try:
            with Image.open(file.stream) as img:
                if img.mode == 'RGBA':
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg.copy()
                else:
                    img = img.copy()
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid image file: {str(e)}"}), 400

        # Generate high-quality ASCII art!
        ascii_art = image_to_ascii(
            img,
            max_width=max_width,
            contrast=contrast,
            brightness=brightness,
            invert=invert,
            detail=detail,
            edge_detection=edge_detection,
            dithering=dithering
        )

        lines = ascii_art.splitlines()
        stats = {
            "characters": len(ascii_art.replace('\n', '')),
            "lines": len(lines),
            "width": len(lines[0]) if lines else 0
        }

        return jsonify({
            "success": True,
            "ascii_art": ascii_art,
            "stats": stats
        })

    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Conversion failed: {str(e)} 💥"
        }), 500


@app.route('/download', methods=['POST'])
def download_txt():
    """Download ASCII art as .txt file"""
    try:
        data = request.get_json(silent=True) or {}
        ascii_art = data.get('ascii_art', '')

        if not ascii_art:
            return jsonify({"error": "No ASCII art to download! 😢"}), 400

        output = io.BytesIO()
        output.write(ascii_art.encode('utf-8'))
        output.seek(0)

        return send_file(
            output,
            mimetype='text/plain',
            as_attachment=False,
            download_name='ascii_art.txt'
        )

    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)} 💥"}), 500


if __name__ == '__main__':
    banner = """
============================================================
🎨 TXTfier Backend Server v2.3 - HIGH QUALITY + SMART! 💎✨
============================================================
💎 FULL QUALITY preserved during conversion!
⚡ Memory optimized where it matters
🎯 Smart downscaling (analysis only)
🔥 float32 instead of float64 (50% memory savings)
✨ Hybrid dithering (quality + speed)
🧠 Light Gaussian blur (detail preservation)
📊 Character Sets: Standard, Extended, Detailed, Blocks
🚀 Best of both worlds: Quality + Efficiency!
============================================================
"""
    print(banner)
    # Use debug=False in production!
    app.run(debug=True, host='0.0.0.0', port=5000)
