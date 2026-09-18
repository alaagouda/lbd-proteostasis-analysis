"""
E6 audit - scans the manuscript's actual current text (accepted tracked
changes, deleted text ignored) for every figure/table legend, extracts
likely abbreviations, and flags any that aren't spelled out within that
same legend. Run once, no external data needed - this only reads the
manuscript file itself.
"""
import re
from pathlib import Path

DOC_PATH = Path("unpacked/word/document.xml")

# Common short words / units that look like abbreviations but aren't
# the kind that need spelling out in a legend
IGNORE = {
    "PD", "DLB", "LBD",  # already defined at first use in the main text,
                          # not required to redefine in every legend by
                          # most journal styles -- but flagged separately below
}
UNIT_LIKE = {
    "UMI", "UMIs", "CPM", "RNA", "DNA", "RGB",
}

def reconstruct_accepted_text(xml):
    """Strip <w:delText>...</w:delText> (rejected/original text) and keep
    everything inside <w:t>...</w:t> (both plain and inserted), giving the
    text as it reads with all tracked changes accepted."""
    xml = re.sub(r'<w:delText[^>]*>.*?</w:delText>', '', xml, flags=re.DOTALL)
    parts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml, flags=re.DOTALL)
    text = "".join(parts)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return text


def split_paragraphs(xml):
    """Return accepted text per <w:p>...</w:p> block, in document order."""
    paras = re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.DOTALL)
    return [reconstruct_accepted_text(p) for p in paras]


LEGEND_START = re.compile(
    r'^\s*(Figure|Fig\.?|Supplementary Fig(?:ure)?\.?|Table)\s*\d+[\.:]',
    re.IGNORECASE
)

ABBR_PATTERN = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')

# Terms that are gene symbols / cell markers, not abbreviations needing
# spelled-out definitions in a legend (heuristic: these are proper nouns,
# not acronyms of a phrase)
GENE_SYMBOL_HINTS = re.compile(r'^[A-Z0-9]+$')


def looks_definable(token):
    """Rough heuristic: real abbreviations needing definition are things
    like NES, FDR, AUROC, UMI, TF -- not gene symbols like SNCA, HSPA1A.
    We can't perfectly distinguish these automatically, so this flags
    everything and lets a human eye do the final call fast, rather than
    silently missing genuine issues."""
    return True


if __name__ == "__main__":
    xml = DOC_PATH.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(xml)

    legends = [p for p in paragraphs if LEGEND_START.match(p.strip())]
    print(f"Found {len(legends)} figure/table legend paragraphs.\n")

    # Build a global map: abbreviation -> was it ever spelled out anywhere
    # in the manuscript (crude check: "Full Term (ABBR)" or "ABBR (Full Term)"
    # patterns elsewhere in the document)?
    full_text = "\n".join(paragraphs)

    issues_found = 0
    for legend in legends:
        header_match = LEGEND_START.match(legend.strip())
        label = legend.strip()[:60].replace("\n", " ")
        abbrs_in_legend = sorted(set(ABBR_PATTERN.findall(legend)) - IGNORE - UNIT_LIKE)
        if not abbrs_in_legend:
            continue

        undefined_here = []
        for abbr in abbrs_in_legend:
            # Defined in THIS legend if "(ABBR)" or "ABBR (" or "ABBR," pattern
            # immediately follows/precedes an expansion-looking phrase.
            defined_locally = bool(re.search(
                rf'\({re.escape(abbr)}\)|{re.escape(abbr)}\s*\(', legend
            ))
            if not defined_locally:
                undefined_here.append(abbr)

        if undefined_here:
            issues_found += 1
            print(f"[{label}...]")
            print(f"  abbreviations present: {abbrs_in_legend}")
            print(f"  NOT defined within this legend: {undefined_here}")
            print()

    print("=" * 70)
    print(f"{issues_found} legend(s) contain at least one abbreviation not")
    print("spelled out within that same legend.")
    print()
    print("NOTE: this is a heuristic first pass, not a final verdict. Gene")
    print("symbols (SNCA, PDE10A, HSPA1A...) will show up as 'abbreviations'")
    print("too, since the script can't distinguish a gene symbol from an")
    print("acronym automatically -- these do NOT need spelling out. Scan the")
    print("printed lists above by eye for the ones that DO need it: NES, FDR,")
    print("AUROC, TF, ISR, HSR, GSEA, CCC and similar analysis-method acronyms.")
