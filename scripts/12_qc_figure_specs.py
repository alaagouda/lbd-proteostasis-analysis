"""
E4 / C7 - checks every figure image file against typical journal specs
for size, resolution, and file weight, and flags outliers so you can see
at a glance which figures are oversized, undersized, or too dense/light
relative to the others (C7's "figures vary widely in size" complaint).

Run this in the folder containing your final figure files (PNG/TIFF/PDF).
Adjust SPECS below to your actual journal's author guidelines if these
placeholder values (typical for a Brain Communications-style journal)
don't match exactly -- check your own instructions-for-authors page.
"""
from pathlib import Path
from PIL import Image
import subprocess

FIGURE_DIR = Path(".")  # run from the folder with your figure files
SPECS = {
    "min_dpi": 300,
    "single_column_mm": 85,
    "double_column_mm": 180,
    "max_file_size_mb": 50,
}
MM_PER_INCH = 25.4


def get_image_info(path):
    if path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        try:
            with Image.open(path) as img:
                dpi = img.info.get("dpi", (None, None))
                width_px, height_px = img.size
        except Exception as e:
            return {"unreadable": str(e)}
        dpi_x = dpi[0] if dpi[0] else None
        width_mm = (width_px / dpi_x * MM_PER_INCH) if dpi_x else None
        height_mm = (height_px / dpi_x * MM_PER_INCH) if dpi_x else None
        return {"width_px": width_px, "height_px": height_px,
                "dpi": dpi_x, "width_mm": width_mm, "height_mm": height_mm}
    elif path.suffix.lower() == ".pdf":
        try:
            result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if line.startswith("Page size"):
                    return {"raw_pdfinfo": line.strip()}
        except FileNotFoundError:
            return {"note": "pdfinfo not available; install poppler-utils to inspect PDFs"}
    return {}


if __name__ == "__main__":
    files = sorted([p for p in FIGURE_DIR.glob("*")
                     if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".pdf")])
    if not files:
        print(f"No image files found in {FIGURE_DIR.resolve()} -- run this from the folder "
              f"containing your final figure exports.")
        raise SystemExit

    print(f"{'File':<45} {'Size (px)':<16} {'DPI':<8} {'Size (mm)':<18} {'Weight':<10} Flags")
    print("-" * 120)

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        info = get_image_info(f)
        flags = []

        if "raw_pdfinfo" in info:
            print(f"{f.name:<45} {'(pdf)':<16} {'':<8} {info['raw_pdfinfo']:<18} {size_mb:.1f}MB")
            continue
        if "unreadable" in info:
            print(f"{f.name:<45} UNREADABLE/CORRUPT FILE ({size_mb:.3f}MB on disk) -- {info['unreadable']}")
            continue
        if not info:
            print(f"{f.name:<45} could not read")
            continue

        dpi = info.get("dpi")
        w_mm, h_mm = info.get("width_mm"), info.get("height_mm")
        DPI_TOLERANCE = 0.5  # avoid flagging harmless float rounding, e.g. 299.9994 vs 300

        if dpi is None:
            flags.append("NO DPI METADATA -- cannot verify resolution")
        elif dpi < SPECS["min_dpi"] - DPI_TOLERANCE:
            flags.append(f"DPI {dpi:.0f} < required {SPECS['min_dpi']}")

        if w_mm:
            if w_mm > SPECS["double_column_mm"] * 1.05:
                flags.append(f"width {w_mm:.0f}mm exceeds double-column max {SPECS['double_column_mm']}mm")
            elif SPECS["single_column_mm"] < w_mm < SPECS["double_column_mm"] * 0.9:
                flags.append("width doesn't cleanly match single- or double-column -- check layout")

        if size_mb > SPECS["max_file_size_mb"]:
            flags.append(f"file size {size_mb:.1f}MB exceeds {SPECS['max_file_size_mb']}MB limit")

        size_str = f"{info['width_px']}x{info['height_px']}"
        dpi_str = f"{dpi:.0f}" if dpi else "?"
        mm_str = f"{w_mm:.0f}x{h_mm:.0f}" if w_mm else "?"
        flag_str = "; ".join(flags) if flags else "OK"
        print(f"{f.name:<45} {size_str:<16} {dpi_str:<8} {mm_str:<18} {size_mb:.1f}MB   {flag_str}")

    print("\nFor C7 (figures vary widely in size): compare the 'Size (mm)' column across all")
    print("figures above -- main figures should generally cluster around one or two consistent")
    print("widths (single-column ~85mm or double-column ~180mm), not scattered arbitrary sizes.")
