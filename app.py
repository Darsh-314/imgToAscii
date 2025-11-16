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

# OPTIMIZED character sets - these actually look GOOD! 🎨
# Sorted from darkest to lightest based on visual density

# Standard ASCII (simple but effective)
ASCII_STANDARD = " .:-=+*#%@$>"

# Extended ASCII (more detail)
ASCII_EXTENDED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Block characters (super clean for photos)
ASCII_BLOCKS = " ░▒▓█"

# Detailed set (best quality)
ASCII_DETAILED = " .`-_':,;^=+/\"|)\\<>)iv%xclrs{*}I?!][1taeo7zjLunT#JCwfy325Fh9kpq4VWXKAmOg6PdQDbUG8RNM@"

def analyze_image(img):
    """
    SIMPLE and EFFECTIVE auto-settings! 🎯
    Stop overthinking, just make it look good!
    """
    gray = img.convert("L")
    pixels = np.array(gray, dtype=np.float64)

    # Basic stats
    mean = float(pixels.mean())
    std = float(pixels.std())
    
    # Key percentiles
    p10, p50, p90 = np.percentile(pixels, [10, 50, 90])
    
    # Simple metrics
    dynamic_range = float((p90 - p10) / 255.0)
    texture = float(std / 255.0)
    
    # Edge detection
    sobel_x = ndimage.sobel(pixels, axis=1)
    sobel_y = ndimage.sobel(pixels, axis=0)
    edges = np.hypot(sobel_x, sobel_y)
    edge_score = float(edges.mean() / 50.0)
    
    # How many pixels are bright?
    bright_ratio = float((pixels > 180).sum() / pixels.size)
    
    settings = {}
    
    # ==========================================
    # CONTRAST - Adjust based on image brightness!
    # ==========================================
    # Lower contrast for dark images to prevent crushing!
    if mean < 85:
        # Dark image - reduce contrast to preserve detail
        settings['contrast'] = 1.0
    elif dynamic_range < 0.35:
        settings['contrast'] = 2.2      # Flat image
    elif dynamic_range < 0.5:
        settings['contrast'] = 1.7      # Low contrast
    elif dynamic_range > 0.75:
        settings['contrast'] = 1.2      # Already good - still boost a bit
    else:
        settings['contrast'] = 1.5      # Normal - needs punch!
    
    # ==========================================
    # BRIGHTNESS - Be MORE aggressive!
    # ==========================================
    # If lots of bright pixels OR high mean = pull down HARDER
    
    if bright_ratio > 0.25 or mean > 145:
        # Bright image - STRONG reduction!
        if mean > 155 or bright_ratio > 0.35:
            settings['brightness'] = -25
        else:
            settings['brightness'] = -20
    elif mean < 90:
        # Dark image - increase
        if mean < 70:
            settings['brightness'] = 30
        else:
            settings['brightness'] = 18
    elif mean < 105:
        # Slightly dark
        settings['brightness'] = 10
    elif mean > 135:
        # Slightly bright - be more aggressive
        settings['brightness'] = -12
    else:
        # Just right!
        settings['brightness'] = 0
    
    # ==========================================
    # EDGE DETECTION - Keep it MINIMAL! (0-0.2 range)
    # ==========================================
    if edge_score < 0.2:
        settings['edge_detection'] = 0.2    # Very soft - gentle boost
    elif edge_score < 0.5:
        settings['edge_detection'] = 0.15   # Normal - minimal
    elif edge_score > 1.0:
        settings['edge_detection'] = 0.0    # Already sharp - none!
    else:
        settings['edge_detection'] = 0.1    # Default minimal
    
    # ==========================================
    # DETAIL LEVEL - ALWAYS use simple/low!
    # ==========================================
    # Simple char set looks cleanest for most images!
    settings['detail'] = 'low'  # ALWAYS simple/standard set!
    
    # ==========================================
    # DITHERING - ALWAYS ON!
    # ==========================================
    # Dithering smooths gradients and looks better!
    settings['dithering'] = True  # ALWAYS dither!
    
    # ==========================================
    # INVERT - Only if REALLY dark
    # ==========================================
    settings['invert'] = bool(mean < 60)
    
    # ==========================================
    # GAMMA - Subtle midtone adjustment
    # ==========================================
    if mean < 100:
        gamma = 0.85    # Lift shadows
    elif mean > 150:
        gamma = 1.15    # Compress highlights
    else:
        gamma = 1.0     # Neutral
    
    settings['gamma'] = float(gamma)
    
    # ==========================================
    # ANALYSIS - Keep it minimal
    # ==========================================
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
    """Apply improved Sobel-based edge detection with noise reduction and better contrast! ✨"""
    if strength == 0:
        return img

    # Convert to numpy
    arr = np.array(img).astype(np.float32)

    # OPTIONAL: Pre-blur slightly to reduce noise
    # (This dramatically improves edge clarity without losing detail)
    smoothed = ndimage.gaussian_filter(arr, sigma=0.7)

    # Sobel filters
    sobel_x = ndimage.sobel(smoothed, axis=1)
    sobel_y = ndimage.sobel(smoothed, axis=0)

    # Edge magnitude
    edges = np.hypot(sobel_x, sobel_y)

    # Avoid divide-by-zero
    max_val = edges.max()
    if max_val == 0:
        edges[:] = 0
    else:
        # Normalize edges to 0–255
        edges = (edges / max_val) * 255.0

    # Apply a non-linear boost curve — makes edges POP ✨
    # (Soft curve: raises medium edges without blowing highlights)
    edges = np.sqrt(edges / 255.0) * 255.0

    edges = edges.astype(np.uint8)

    # Blend with original
    if strength < 1.0:
        blended = (arr * (1 - strength) + edges * strength).astype(np.uint8)
    else:
        blended = edges

    return Image.fromarray(blended)


def apply_dithering(img):
    """Apply improved Floyd-Steinberg dithering for retro texture! 🎨"""
    arr = np.array(img, dtype=np.float32)
    height, width = arr.shape

    # Floyd-Steinberg error diffusion
    for y in range(height):
        for x in range(width):
            old_pixel = arr[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            arr[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Right
            if x + 1 < width:
                arr[y, x + 1] += error * 7 / 16
            # Bottom-left
            if y + 1 < height:
                if x > 0:
                    arr[y + 1, x - 1] += error * 3 / 16
                # Bottom
                arr[y + 1, x] += error * 5 / 16
                # Bottom-right
                if x + 1 < width:
                    arr[y + 1, x + 1] += error * 1 / 16

    # Clip values to valid range before converting
    arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))

    
    return Image.fromarray(arr.astype(np.uint8))

def apply_image_enhancements(img, contrast=1.0, brightness=0, sharpness=1.0):
    """Apply enhancements to make ASCII look BETTER! ✨ (optimized, safe)"""
    
    # --- Contrast ---
    if contrast != 1.0:
        # Clamp contrast to avoid crazy values
        contrast = max(0.0, contrast)
        img = ImageEnhance.Contrast(img).enhance(contrast)
    
    # --- Brightness ---
    if brightness != 0:
        # Convert brightness (-100 to 100) to multiplier safely
        brightness_factor = 1.0 + max(-1.0, min(brightness / 100.0, 10.0))
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    
    # --- Sharpness ---
    if sharpness != 1.0:
        # Clamp sharpness to reasonable range
        sharpness = max(0.0, sharpness)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    return img


def get_char_set(detail_level='high'):
    """Get the right character set based on detail level! 🎯"""
    sets = {
        'low': ASCII_STANDARD,
        'medium': ASCII_EXTENDED,
        'high': ASCII_DETAILED,
        'blocks': ASCII_BLOCKS
    }
    return sets.get(detail_level, ASCII_DETAILED)

def pixel_to_char(pixel_value, char_set, invert=False, gamma=1.0):
    """Convert a pixel brightness to the perfect character! 🎨 (optimized)"""
    
    # Clamp pixel to valid range
    pixel_value = max(0, min(pixel_value, 255))
    
    # Invert if needed
    if invert:
        pixel_value = 255 - pixel_value
    
    # Normalize and apply gamma correction
    normalized = pixel_value / 255.0
    gamma_corrected = pow(normalized, gamma)
    
    # Map to character set
    char_index = int(gamma_corrected * (len(char_set) - 1))
    
    # Clamp to valid index
    char_index = max(0, min(char_index, len(char_set) - 1))
    
    return char_set[char_index]


def image_to_ascii(
    img, max_width=100, contrast=1.0, brightness=0, invert=False, detail='high', 
    edge_detection=0.0, dithering=False
):
    """Convert image to GORGEOUS ASCII art with CUSTOM options! 🔥 (optimized)"""
    
    # --- Grayscale conversion ---
    img = img.convert("L")
    
    # --- Apply image enhancements ---
    img = apply_image_enhancements(img, contrast, brightness, sharpness=1.5)
    
    # --- Apply edge detection ---
    if edge_detection > 0:
        img = apply_edge_detection(img, edge_detection)
    
    # --- Apply dithering ---
    if dithering:
        img = apply_dithering(img)
    
    # --- Calculate new dimensions while keeping ASCII aspect ratio ---
    aspect_ratio = img.height / img.width
    new_height = max(1, int(max_width * aspect_ratio * 0.55))  # 0.55 = char aspect compensation
    
    # --- Resize with high-quality resampling ---
    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # --- Convert to NumPy array once for efficiency ---
    pixels = np.array(img, dtype=np.uint8)
    
    # --- Select character set ---
    char_set = get_char_set(detail)
    
    # --- Generate ASCII lines ---
    ascii_lines = [
        ''.join(pixel_to_char(pixel, char_set, invert) for pixel in row)
        for row in pixels
    ]
    
    # --- Return as single string ---
    return '\n'.join(ascii_lines)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint! 💚"""
    return jsonify({
        "status": "alive",
        "message": "TXTfier backend is RUNNING! 🚀",
        "version": "2.1 - AUTO SETTINGS! 🧠"
    })


@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    """Analyze image and suggest optimal settings! 🧠✨"""
    try:
        # --- Validate file upload ---
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image uploaded!" if not file else "Empty filename!"
            }), 400

        # --- Open image safely ---
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

        # --- Analyze image ---
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
    """Main conversion endpoint with IMPROVED quality! 🎨"""
    try:
        # --- Validate file upload ---
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image uploaded!" if not file else "Empty filename! 😢"
            }), 400

        # --- Get parameters with validation ---
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

        # --- Open and validate image ---
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

        # --- Generate ASCII art ---
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

        # --- Calculate stats ---
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
    """Download ASCII art as .txt file! 📥"""
    try:
        data = request.get_json(silent=True) or {}
        ascii_art = data.get('ascii_art', '')

        if not ascii_art:
            return jsonify({"error": "No ASCII art to download! 😢"}), 400

        # --- Create file in memory ---
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
🎨 TXTfier Backend Server v2.1 - AUTO SETTINGS! 🧠✨
============================================================
📊 Character Sets: Standard, Extended, Detailed, Blocks
✨ Edge Detection: Make outlines POP!
🎨 Dithering: Retro texture effects!
🧠 AUTO ANALYSIS: Smart settings detection!
⚡ Sharpening and enhancement enabled!
🔥 Ready to create BEAUTIFUL ASCII art!
============================================================
"""
    print(banner)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)


