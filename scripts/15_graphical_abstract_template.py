"""
E9 -- graphical abstract template, built to the journal's stated spec
(120x120mm, 300-600dpi, Arial 12-16pt, per this project's Editor comments).

This does NOT create your final graphical abstract -- it creates a
correctly-sized, correctly-calibrated blank canvas with guide boxes and
placeholder text at the right font sizes, so you (or a designer) can drop
in the actual schematic content without guessing at dimensions. Open the
output PNG in any image editor and replace the placeholder boxes with
real content, keeping the canvas size and DPI unchanged.
"""
from PIL import Image, ImageDraw, ImageFont

DPI = 600  # use the higher end of the allowed 300-600 range for print quality
SIZE_MM = 120
MM_PER_INCH = 25.4
SIZE_PX = int(SIZE_MM / MM_PER_INCH * DPI)

FONT_SIZES_PT = {"title": 16, "body": 12}


def pt_to_px(pt, dpi):
    return int(pt / 72 * dpi)


def try_load_font(size_px):
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size_px)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    img = Image.new("RGB", (SIZE_PX, SIZE_PX), "white")
    draw = ImageDraw.Draw(img)

    margin = int(SIZE_PX * 0.04)
    draw.rectangle([margin, margin, SIZE_PX - margin, SIZE_PX - margin],
                   outline="gray", width=2)

    title_font = try_load_font(pt_to_px(FONT_SIZES_PT["title"], DPI))
    body_font = try_load_font(pt_to_px(FONT_SIZES_PT["body"], DPI))

    draw.text((margin + 10, margin + 10),
              "TITLE PLACEHOLDER", fill="black", font=title_font)

    # Three placeholder panels for a typical three-cohort schematic
    panel_w = (SIZE_PX - 2 * margin - 40) // 3
    panel_top = int(SIZE_PX * 0.25)
    panel_h = int(SIZE_PX * 0.45)
    labels = ["Jin\n(temporal)", "Ma\n(cingulate)", "Nido\n(prefrontal)"]
    for i, label in enumerate(labels):
        x0 = margin + 20 + i * (panel_w + 10)
        x1 = x0 + panel_w
        draw.rectangle([x0, panel_top, x1, panel_top + panel_h],
                       outline="steelblue", width=2)
        line_height = pt_to_px(FONT_SIZES_PT["body"], DPI) + 8
        for j, line in enumerate(label.split("\n")):
            draw.text((x0 + 10, panel_top + 10 + j * line_height),
                      line, fill="steelblue", font=body_font)

    finding_y = panel_top + panel_h + 30
    finding_lines = [
        "KEY FINDING PLACEHOLDER (Arial 12pt)",
        "e.g. arrow/gradient showing proteostasis",
        "effect size decreasing left to right",
    ]
    line_height = pt_to_px(FONT_SIZES_PT["body"], DPI) + 10
    for j, line in enumerate(finding_lines):
        draw.text((margin + 10, finding_y + j * line_height),
                  line, fill="black", font=body_font)

    out_path = "graphical_abstract_template.png"
    img.save(out_path, dpi=(DPI, DPI))
    print(f"Saved {out_path}: {SIZE_PX}x{SIZE_PX}px at {DPI}dpi ({SIZE_MM}x{SIZE_MM}mm)")
    print("Open this file, replace the placeholder boxes/text with your real schematic,")
    print("and keep the canvas size and DPI exactly as generated.")
