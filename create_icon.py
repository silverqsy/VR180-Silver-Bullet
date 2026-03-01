#!/usr/bin/env python3
"""
Generate VR180 Silver Bullet app icon
Creates a sleek, modern icon with VR180 and bullet imagery
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_icon(size):
    """Create a single icon at the specified size"""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Color scheme: Silver/Chrome gradient with blue accent
    silver_light = (220, 220, 235)
    silver_dark = (140, 140, 160)
    blue_accent = (30, 120, 220)
    blue_bright = (60, 160, 255)

    # Calculate dimensions
    margin = size * 0.1
    center_x = size // 2
    center_y = size // 2

    # Draw outer circle (background) - gradient effect with multiple circles
    for i in range(20):
        alpha = int(255 * (1 - i/20))
        radius = size * 0.45 * (1 - i/40)
        color = tuple([int(silver_light[j] * (1 - i/20) + silver_dark[j] * (i/20)) for j in range(3)] + [alpha])
        draw.ellipse([center_x - radius, center_y - radius,
                     center_x + radius, center_y + radius],
                    fill=color)

    # Draw main circle with gradient
    main_radius = size * 0.42
    draw.ellipse([center_x - main_radius, center_y - main_radius,
                 center_x + main_radius, center_y + main_radius],
                fill=(*silver_dark, 255))

    # Draw inner circle (lighter)
    inner_radius = size * 0.38
    draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                 center_x + inner_radius, center_y + inner_radius],
                fill=(*silver_light, 255))

    # Draw VR180 representation - two hemispheres (left/right eyes)
    hemisphere_radius = size * 0.15
    left_eye_x = center_x - size * 0.12
    right_eye_x = center_x + size * 0.12
    eye_y = center_y - size * 0.05

    # Left hemisphere (blue gradient)
    for i in range(10):
        r = hemisphere_radius * (1 - i/20)
        alpha = int(255 * (1 - i/15))
        color = tuple([int(blue_accent[j] * (1 - i/10) + blue_bright[j] * (i/10)) for j in range(3)] + [alpha])
        draw.ellipse([left_eye_x - r, eye_y - r,
                     left_eye_x + r, eye_y + r],
                    fill=color)

    # Right hemisphere (blue gradient)
    for i in range(10):
        r = hemisphere_radius * (1 - i/20)
        alpha = int(255 * (1 - i/15))
        color = tuple([int(blue_accent[j] * (1 - i/10) + blue_bright[j] * (i/10)) for j in range(3)] + [alpha])
        draw.ellipse([right_eye_x - r, eye_y - r,
                     right_eye_x + r, eye_y + r],
                    fill=color)

    # Draw bullet/projectile shape pointing right (representing "Silver Bullet")
    bullet_y = center_y + size * 0.15
    bullet_length = size * 0.35
    bullet_width = size * 0.08

    # Bullet body (rectangle with rounded end)
    bullet_points = [
        (center_x - bullet_length/2, bullet_y - bullet_width/2),
        (center_x + bullet_length/2 - bullet_width/2, bullet_y - bullet_width/2),
        (center_x + bullet_length/2, bullet_y),
        (center_x + bullet_length/2 - bullet_width/2, bullet_y + bullet_width/2),
        (center_x - bullet_length/2, bullet_y + bullet_width/2),
    ]

    # Draw bullet with silver gradient
    draw.polygon(bullet_points, fill=(*silver_dark, 255))

    # Add highlight to bullet
    highlight_points = [
        (center_x - bullet_length/2 + 5, bullet_y - bullet_width/2 + 2),
        (center_x + bullet_length/3, bullet_y - bullet_width/2 + 2),
        (center_x + bullet_length/3, bullet_y - bullet_width/2 + 5),
        (center_x - bullet_length/2 + 5, bullet_y - bullet_width/2 + 5),
    ]
    draw.polygon(highlight_points, fill=(255, 255, 255, 180))

    # Draw bullet tip (pointed end)
    tip_points = [
        (center_x + bullet_length/2 - bullet_width/2, bullet_y - bullet_width/2),
        (center_x + bullet_length/2, bullet_y),
        (center_x + bullet_length/2 - bullet_width/2, bullet_y + bullet_width/2),
    ]
    draw.polygon(tip_points, fill=(*blue_bright, 255))

    # Add "180" text if size is large enough
    if size >= 256:
        try:
            # Try to use system font, fall back to default if not available
            font_size = int(size * 0.12)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()

            text = "180"
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = center_x - text_width // 2
            text_y = center_y + size * 0.28

            # Draw text with shadow
            draw.text((text_x + 2, text_y + 2), text, font=font, fill=(0, 0, 0, 100))
            draw.text((text_x, text_y), text, font=font, fill=(*blue_bright, 255))
        except:
            pass  # Skip text if font rendering fails

    return img

def main():
    """Generate all required icon sizes"""
    print("Generating VR180 Silver Bullet icon...")

    # macOS requires these sizes for .icns
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    for size in sizes:
        img = create_icon(size)
        filename = f"icon_{size}x{size}.png"
        img.save(filename)
        print(f"✓ Created {filename}")

    print("\nIcon generation complete!")
    print("Run './create_icns.sh' to convert to .icns format")

if __name__ == "__main__":
    main()
