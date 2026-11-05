"""Re-run the STRING query for the 14 proteostasis genes with every setting written down, and re-check the Tier 1 edges.

Why this exists
---------------
The STRING network in Supplementary Fig. 2 was made by an earlier query whose settings were not recorded. This script repeats the query
in the open: it names the STRING release, asks for both the "functional" and the "physical" network at a minimum combined score of 0.40,
saves the raw answers, and counts how many of the 60 Tier 1 co-expression edges each network contains. It also compares the fresh
networks with your stored file (Ma_PPI_network.csv), so you can see which network type that file really came from.

What it reads (from data_snrna/ on disk, or from your Google Drive folder if it is not there)
  Ma_Correlation_network.csv   the 91 gene pairs with rho, padj and Tier (written by gene_network.py)
  Ma_PPI_network.csv           optional: the stored STRING edges (Gene1, Gene2, Combined_score)

What it writes (to results/string_rerun/, and to a "string_rerun" folder in Google Drive when a token is available)
  string_rerun_parameters.json      every setting, the STRING release, the date, package versions
  STRING_functional_raw.json        the answer from STRING, exactly as received (same for physical)
  STRING_functional_edges.csv       the edges among the 14 genes, with the evidence scores (same for physical)
  Tier1_edge_check.csv              one row per Tier 1 edge: is it in each network, with which score
  STRING_rerun_report.md            the numbers, a comparison with the manuscript, and sentences you can adapt

How to run
  python string_network_query.py --folder-id YOUR_DRIVE_FOLDER_ID       reads and writes Google Drive
  python string_network_query.py --local-only                           uses only files on this computer

Google Drive needs a login file named drive_token.json (see the README). A token that can only read Drive still works: the results then
stay on this computer. The folder id can also be set with the environment variable LBD_DRIVE_FOLDER_ID.
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------------------------------------------------
# Settings. Everything the query depends on is here, so nothing is hidden in the code below.
# ---------------------------------------------------------------------------------------------------------------------
GENES = [
    "HSPA1A",
    "HSPA1B",
    "HSPB1",
    "DNAJB1",
    "DNAJB4",
    "HSP90AA1",
    "HSP90AB1",
    "DNAJA1",
    "DNAJA4",
    "CHORDC1",
    "STIP1",
    "CRYAB",
    "BAG3",
    "HSPH1",
]
SPECIES = 9606  # Homo sapiens
REQUIRED_SCORE = 400  # STRING scores run from 0 to 1000, so 400 means a combined score of 0.40
NETWORK_TYPES = ["functional", "physical"]
STRING_HOST = "https://version-12-0.string-db.org"  # the address of release 12.0; set to None to use whichever release is current
EXPECTED_VERSION = "12.0"  # the release named in the Methods
CALLER_IDENTITY = "lbd_proteostasis_analysis"  # STRING asks every program to name itself

DRIVE_FOLDER_ID = os.environ.get(
    "LBD_DRIVE_FOLDER_ID", ""
)  # the Google Drive folder that holds your input files (the long code at the end of its web address)
CORRELATION_FILE = "Ma_Correlation_network.csv"
STORED_STRING_FILE = "Ma_PPI_network.csv"
OUTPUT_FOLDER = "string_rerun"

# Numbers stated in the manuscript. They are only used to say whether the new run agrees; they never change the result.
IN_MANUSCRIPT = {"pairs": 91, "tier1": 60, "supported": 53, "string_edges": 80}
TIER1_RULE = "padj < 0.01 and |rho| > 0.70"


# ---------------------------------------------------------------------------------------------------------------------
# Step 1. Read the correlation table and pick out the Tier 1 edges.
# ---------------------------------------------------------------------------------------------------------------------
def make_pair(a, b):
    """An edge has no direction, so (A, B) and (B, A) must count as the same edge: sort the two names."""
    a, b = str(a).strip().upper(), str(b).strip().upper()
    if a == b:
        raise ValueError(f"A gene cannot pair with itself: {a}")
    return tuple(sorted((a, b)))


def find_column(table, *keywords):
    """Return the first column whose name contains one of the keywords (case does not matter), or None."""
    for column in table.columns:
        if any(word in column.lower() for word in keywords):
            return column
    return None


def load_tier1(path):
    """Read the 91-pair table. Returns one row per pair with a True/False column 'tier1'."""
    table = pd.read_csv(path)
    col1, col2 = find_column(table, "gene1", "gene_a"), find_column(table, "gene2", "gene_b")
    rho, padj, tier = (
        find_column(table, "rho", "spearman"),
        find_column(table, "padj", "fdr", "qvalue"),
        find_column(table, "tier"),
    )
    if col1 is None or col2 is None:
        raise ValueError(
            f"Cannot find the two gene columns in {path}. Columns present: {list(table.columns)}"
        )
    pairs = [make_pair(a, b) for a, b in zip(table[col1], table[col2])]
    if len(pairs) != len(set(pairs)):
        raise ValueError("The correlation table lists the same pair of genes more than once.")
    result = pd.DataFrame({"gene1": [p[0] for p in pairs], "gene2": [p[1] for p in pairs]})
    if rho is not None:
        result["rho"] = table[rho].values
    if padj is not None:
        result["padj"] = table[padj].values
    if tier is not None:  # the table already says which pairs are Tier 1
        result["tier1"] = table[tier].astype(str).str.startswith("Tier 1").values
    elif rho is not None and padj is not None:  # otherwise apply the rule from the Methods
        result["tier1"] = ((table[padj] < 0.01) & (table[rho].abs() > 0.70)).values
    else:
        raise ValueError(
            "The table has no Tier column and no rho and padj columns, so Tier 1 cannot be worked out."
        )
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Step 2. Ask STRING. Every request goes through this small client, which keeps the exact address and the exact answer.
# ---------------------------------------------------------------------------------------------------------------------
class StringClient:
    def __init__(self, host, session=None, pause=2.0):
        self.host = host.rstrip("/")
        self.session = session or requests
        self.pause = pause  # seconds to wait before a repeat attempt

    def get(self, path, params, host=None):
        """Return (decoded answer, raw text, final web address). Tries three times before giving up."""
        url = f"{(host or self.host).rstrip('/')}/api/json/{path}"
        problem = ""
        for attempt in (1, 2, 3):
            try:
                reply = self.session.get(url, params=params, timeout=60)
                if reply.status_code == 200:
                    return reply.json(), reply.text, getattr(reply, "url", url)
                problem = f"HTTP {reply.status_code}: {reply.text[:200]}"
            except requests.RequestException as error:
                problem = str(error)
            time.sleep(self.pause * attempt)
        raise RuntimeError(
            f"STRING did not answer {url} after three tries. Last problem: {problem}"
        )

    def version(self):
        """Ask the server which STRING release it is serving."""
        answer, raw, url = self.get("version", {"caller_identity": CALLER_IDENTITY})
        info = answer[0] if isinstance(answer, list) else answer
        return {
            "string_version": info.get("string_version"),
            "stable_address": info.get("stable_address"),
            "asked": url,
            "raw_answer": raw[:300],
        }

    def resolve(self, genes):
        """Check how STRING understands each gene name. Returns {query name: STRING's row for it}."""
        params = {
            "identifiers": "\r".join(genes),
            "species": SPECIES,
            "limit": 1,
            "caller_identity": CALLER_IDENTITY,
        }
        rows, _, _ = self.get("get_string_ids", params)
        return {row["queryItem"].upper(): row for row in rows}

    def network(self, genes, network_type, required_score):
        """The interactions among the genes. Returns (rows, raw text, web address)."""
        params = {
            "identifiers": "\r".join(genes),
            "species": SPECIES,
            "required_score": required_score,
            "network_type": network_type,
            "add_nodes": 0,  # do not add other proteins: we only want edges among our 14 genes
            "caller_identity": CALLER_IDENTITY,
        }
        return self.get("network", params)


EVIDENCE = [
    "score",
    "escore",
    "dscore",
    "tscore",
    "ascore",
    "fscore",
    "pscore",
    "nscore",
]  # combined, experiments, databases, text mining, ...


def parse_edges(rows, to_query_name):
    """Turn STRING's rows into a table with one row per unordered pair and the highest score if a pair appears twice."""
    records = {}
    for row in rows:
        a = to_query_name.get(row["preferredName_A"].upper(), row["preferredName_A"].upper())
        b = to_query_name.get(row["preferredName_B"].upper(), row["preferredName_B"].upper())
        if a == b:
            continue
        key = make_pair(a, b)
        record = {"gene1": key[0], "gene2": key[1]}
        record.update({name: float(row.get(name, 0.0)) for name in EVIDENCE})
        if key not in records or record["score"] > records[key]["score"]:
            records[key] = record
    columns = ["gene1", "gene2"] + EVIDENCE
    return (
        pd.DataFrame(list(records.values()), columns=columns)
        .sort_values(["gene1", "gene2"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------------------------------------------------
# Step 3. Count how many Tier 1 edges each network contains, and compare with the stored file.
# ---------------------------------------------------------------------------------------------------------------------
def check_tier1(tier1_pairs, edges):
    """For each Tier 1 pair: is it in the network, and what are its combined and experimental scores?"""
    lookup = {(row.gene1, row.gene2): row for row in edges.itertuples()}
    rows = []
    for pair in tier1_pairs.itertuples():
        hit = lookup.get((pair.gene1, pair.gene2))
        rows.append(
            {
                "gene1": pair.gene1,
                "gene2": pair.gene2,
                "in_network": hit is not None,
                "combined_score": hit.score if hit is not None else float("nan"),
                "experimental_score": hit.escore if hit is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_stored(path):
    """Read the stored STRING edges. Scores above 1 are taken to be on STRING's 0-1000 scale and divided by 1000."""
    table = pd.read_csv(path)
    col1, col2 = find_column(table, "gene1", "preferredname_a"), find_column(
        table, "gene2", "preferredname_b"
    )
    score = find_column(table, "combined_score", "score")
    if col1 is None or col2 is None or score is None:
        raise ValueError(
            f"Cannot read the stored STRING file {path}. Columns present: {list(table.columns)}"
        )
    scores = pd.to_numeric(table[score])
    if scores.max() > 1.0:
        scores = scores / 1000.0
    pairs = [make_pair(a, b) for a, b in zip(table[col1], table[col2])]
    return pd.DataFrame(
        {"gene1": [p[0] for p in pairs], "gene2": [p[1] for p in pairs], "score": scores.values}
    )


def compare_with_stored(stored, fresh):
    """How different is the stored edge list from a fresh network? Identical means the same edges with the same scores."""
    old = {(r.gene1, r.gene2): r.score for r in stored.itertuples()}
    new = {(r.gene1, r.gene2): r.score for r in fresh.itertuples()}
    shared = old.keys() & new.keys()
    biggest = max((abs(old[k] - new[k]) for k in shared), default=0.0)
    return {
        "stored_edges": len(old),
        "fresh_edges": len(new),
        "shared_edges": len(shared),
        "only_in_stored": sorted(old.keys() - new.keys()),
        "only_in_fresh": sorted(new.keys() - old.keys()),
        "largest_score_difference": round(biggest, 4),
        "identical": old.keys() == new.keys() and biggest < 0.0015,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Google Drive. Used only if a folder id is given and a login file is found. All Google code is inside this class.
# ---------------------------------------------------------------------------------------------------------------------
FOLDER_TYPE = "application/vnd.google-apps.folder"


def token_places():
    """Where the Drive login file is looked for, in order."""
    return [
        os.environ.get("LBD_DRIVE_TOKEN") or "(LBD_DRIVE_TOKEN is not set)",
        "drive_token.json",
        "/marimo/drive_token.json",
        str(Path.home() / "drive_token.json"),
    ]


def find_token():
    """Look for the Drive login file in the usual places. Returns a path or None."""
    for candidate in token_places():
        if Path(candidate).is_file():
            return candidate
    return None


def _download_file(service, file_id, destination):
    from googleapiclient.http import MediaIoBaseDownload

    with open(destination, "wb") as handle:
        downloader = MediaIoBaseDownload(handle, service.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _media(path):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), resumable=False)


class DriveFolder:
    """A Google Drive folder that we can search (including its sub-folders), download from, and upload to."""

    def __init__(self, folder_id, token_path=None, service=None):
        self.folder_id = folder_id
        self.can_write = True
        if service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            token = json.loads(Path(token_path).read_text(encoding="utf-8"))
            scopes = token.get("scopes") or []
            self.can_write = any(s.endswith(("/auth/drive", "/auth/drive.file")) for s in scopes)
            service = build(
                "drive",
                "v3",
                credentials=Credentials.from_authorized_user_file(token_path),
                cache_discovery=False,
            )
        self.service = service
        self._subfolders = {}

    def _children(self, folder_id):
        """Everything directly inside one folder."""
        rows, page = [], None
        while True:
            answer = (
                self.service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageSize=1000,
                    pageToken=page,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            rows += answer.get("files", [])
            page = answer.get("nextPageToken")
            if not page:
                return rows

    def find(self, name, max_depth=4):
        """Search the folder and its sub-folders for a file with this name. If there are several, the newest wins."""
        matches, queue = [], [(self.folder_id, 0)]
        while queue:
            folder, depth = queue.pop(0)
            for row in self._children(folder):
                if row["name"] == name and row["mimeType"] != FOLDER_TYPE:
                    matches.append(row)
                elif row["mimeType"] == FOLDER_TYPE and depth < max_depth:
                    queue.append((row["id"], depth + 1))
        return max(matches, key=lambda r: r.get("modifiedTime", "")) if matches else None

    def download(self, name, destination):
        """Copy a file from Drive to this computer. Returns True if it was found."""
        row = self.find(name)
        if row is None:
            return False
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        _download_file(self.service, row["id"], destination)
        return True

    def _subfolder(self, name):
        """The sub-folder with this name inside the main folder; it is created the first time."""
        if name not in self._subfolders:
            existing = [
                r
                for r in self._children(self.folder_id)
                if r["name"] == name and r["mimeType"] == FOLDER_TYPE
            ]
            if existing:
                self._subfolders[name] = existing[0]["id"]
            else:
                body = {"name": name, "mimeType": FOLDER_TYPE, "parents": [self.folder_id]}
                self._subfolders[name] = (
                    self.service.files()
                    .create(body=body, fields="id", supportsAllDrives=True)
                    .execute()["id"]
                )
        return self._subfolders[name]

    def upload(self, path, subfolder):
        """Put a local file into the sub-folder. A file of the same name is replaced, not duplicated."""
        parent = self._subfolder(subfolder)
        old = [r for r in self._children(parent) if r["name"] == Path(path).name]
        if old:
            self.service.files().update(
                fileId=old[0]["id"], media_body=_media(path), fields="id", supportsAllDrives=True
            ).execute()
        else:
            body = {"name": Path(path).name, "parents": [parent]}
            self.service.files().create(
                body=body, media_body=_media(path), fields="id", supportsAllDrives=True
            ).execute()


def get_input(name, local_dir, drive, required=True):
    """Find an input file: first in the local folder, then in Google Drive. Returns a path, or None if optional and missing."""
    local = Path(local_dir) / name
    if local.is_file():
        return local
    if drive is not None and drive.download(name, local):
        print(f"  downloaded {name} from Google Drive")
        return local
    if required:
        where = "or in the Google Drive folder" if drive else "and Google Drive is not in use"
        raise FileNotFoundError(
            f"{name} was not found in {local_dir} {where}. Either put the file in {local_dir}, or set DRIVE_FOLDER_ID near the top of "
            "this script (or pass --folder-id) and make sure drive_token.json is in the working folder."
        )
    return None


# ---------------------------------------------------------------------------------------------------------------------
# The report: the numbers, whether they agree with the manuscript, and sentences you can adapt.
# ---------------------------------------------------------------------------------------------------------------------
def build_report(info, results, comparison, n_tier1):
    lines = ["# STRING re-run: settings and results", ""]
    lines += [
        f"Run on {info['retrieved_utc']}. Server queried: {info['string_host_queried']} (reports STRING release {info['string_version_reported']}).",
        f"Species: Homo sapiens (taxon {SPECIES}). Minimum combined score: {REQUIRED_SCORE / 1000:.2f} (STRING scale {REQUIRED_SCORE} of 1000).",
        f"Genes asked about ({len(GENES)}): {', '.join(GENES)}. Tier 1 rule used: {TIER1_RULE}.",
        "",
        "## Results",
        "",
        f"| Network type | Interactions among the {len(GENES)} genes | Tier 1 edges present | Present with experimental evidence | Manuscript says |",
        "|---|---|---|---|---|",
    ]
    for kind, res in results.items():
        said = f"{IN_MANUSCRIPT['string_edges']} interactions, {IN_MANUSCRIPT['supported']} of {IN_MANUSCRIPT['tier1']}"
        lines.append(
            f"| {kind} | {res['n_edges']} | {res['supported']} of {n_tier1} | {res['with_experiments']} | {said} |"
        )
    lines += [
        "",
        "Experimental evidence means a non-zero STRING 'experiments' score for the pair. It is shown because the functional network also counts co-expression and text mining, which are weaker evidence that two proteins really interact.",
        "",
    ]
    for kind, res in results.items():
        agree = (
            res["n_edges"] == IN_MANUSCRIPT["string_edges"]
            and res["supported"] == IN_MANUSCRIPT["supported"]
        )
        lines += [f"### {kind.capitalize()} network", ""]
        lines.append(
            "This **agrees** with the manuscript (80 interactions, 53 of 60)."
            if agree
            else "This **does not match** the manuscript's 80 interactions and 53 of 60."
        )
        absent = res["absent"]
        counts = Counter(g for pair in absent for g in pair).most_common(3)
        lines.append(
            f"Tier 1 edges absent from this network ({len(absent)}): {', '.join('-'.join(p) for p in absent) or 'none'}."
        )
        if counts:
            lines.append(
                f"Genes involved most often in the absent pairs: {', '.join(f'{g} ({n})' for g, n in counts)}."
            )
        lines.append("")
    if comparison:
        lines += ["## Which network type is the stored file?", ""]
        for kind, comp in comparison.items():
            verdict = (
                "IDENTICAL"
                if comp["identical"]
                else f"different ({len(comp['only_in_stored'])} edges only in the stored file, {len(comp['only_in_fresh'])} only in the new network)"
            )
            lines.append(
                f"- Stored file ({comp['stored_edges']} edges) versus new {kind} network ({comp['fresh_edges']} edges): {comp['shared_edges']} edges shared, largest score difference {comp['largest_score_difference']}. {verdict}."
            )
        matched = [kind for kind, comp in comparison.items() if comp["identical"]]
        lines += [
            "",
            (
                f"**Conclusion:** the stored file matches the {' and '.join(matched)} network."
                if matched
                else "**Conclusion:** the stored file matches neither new network, so its origin is still undocumented. Use the new results and replace the stored file."
            ),
            "",
        ]
    lines += [
        "## Sentences to adapt (choose the network type you will report, and check every number)",
        "",
    ]
    for kind, res in results.items():
        lines += [
            f"**If you report the {kind} network.**",
            "",
            (
                f"Methods: STRING (release {info['string_version_reported']}, {info['string_host_queried']}) was queried on {info['retrieved_utc'][:10]} for Homo sapiens with the {len(GENES)} proteostasis genes, "
                f"network type '{kind}', minimum combined score {REQUIRED_SCORE / 1000:.2f}; this returned {res['n_edges']} interactions among the {len(GENES)} genes."
            ),
            "",
            f"Results: {res['supported']} of the {n_tier1} Tier 1 co-expression edges were present in this network, {res['with_experiments']} of them with experimental evidence.",
            "",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# The whole run, step by step.
# ---------------------------------------------------------------------------------------------------------------------
IN_NOTEBOOK = (
    "marimo" in sys.modules or "ipykernel" in sys.modules
)  # a notebook has its own command-line arguments, which must be ignored


def default_workdir():
    """The folder that holds data_snrna/: /marimo in a hosted notebook, otherwise the folder you are in."""
    return "/marimo" if Path("/marimo").is_dir() else "."


def banner(text):
    print(f"\n=== {text}")


def main(argv=None, session=None, pause=2.0):
    parser = argparse.ArgumentParser(
        description="Re-run STRING with explicit settings and re-check the Tier 1 edges."
    )
    parser.add_argument(
        "--workdir",
        default=default_workdir(),
        help="the folder that contains data_snrna/ (results are written under results/)",
    )
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("LBD_DRIVE_FOLDER_ID") or DRIVE_FOLDER_ID,
        help="Google Drive folder with the input files",
    )
    parser.add_argument("--local-only", action="store_true", help="do not use Google Drive at all")
    parser.add_argument(
        "--host",
        default=STRING_HOST,
        help="STRING address; the default is the fixed address of release 12.0",
    )
    args = parser.parse_known_args(argv)[0] if IN_NOTEBOOK else parser.parse_args(argv)
    work = Path(args.workdir)
    input_dir, out_dir = work / "data_snrna", work / "results" / OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("Step 1. Google Drive and input files")
    drive = None
    token = None if args.local_only or not args.folder_id else find_token()
    if args.local_only or not args.folder_id:
        print("  Google Drive is not used (no folder id given, or --local-only).")
    elif token is None:
        print(
            f"  No drive_token.json found (looked in: {'; '.join(token_places())}), so Google Drive is not used."
        )
    else:
        drive = DriveFolder(args.folder_id, token)
        print(
            f"  Google Drive folder {args.folder_id}: {'read and write' if drive.can_write else 'read only; results will stay on this computer'}."
        )
    try:
        correlation_path = get_input(CORRELATION_FILE, input_dir, drive)
        stored_path = get_input(STORED_STRING_FILE, input_dir, drive, required=False)
    except FileNotFoundError as error:
        print(f"  STOPPED: {error}")
        return 3
    print(f"  correlation table: {correlation_path}")
    print(
        f"  stored STRING file: {stored_path if stored_path else 'not found (the comparison with the stored file will be skipped)'}"
    )

    banner("Step 2. Tier 1 edges")
    pairs = load_tier1(correlation_path)
    tier1 = pairs[pairs["tier1"]].reset_index(drop=True)
    print(
        f"  {len(pairs)} pairs in the table (manuscript: {IN_MANUSCRIPT['pairs']}); {len(tier1)} are Tier 1 (manuscript: {IN_MANUSCRIPT['tier1']}) by the rule: {TIER1_RULE}"
    )
    if len(pairs) != IN_MANUSCRIPT["pairs"] or len(tier1) != IN_MANUSCRIPT["tier1"]:
        print(
            "  WARNING: these counts differ from the manuscript; check that this is the Ma table."
        )

    banner("Step 3. STRING release and gene names")
    host = args.host or "https://string-db.org"
    client = StringClient(host, session, pause)
    try:
        version = client.version()
        if args.host and version["string_version"] != EXPECTED_VERSION:
            print(
                f"  STOPPED: {host} reports STRING release {version['string_version']}, but the Methods say {EXPECTED_VERSION}."
            )
            print(f"  STRING's own answer to the version question was: {version['raw_answer']}")
            return 2
        if not args.host:
            client = StringClient(
                version["stable_address"], session, pause
            )  # use the fixed address of the current release so it is recorded
        print(f"  server {client.host} reports STRING release {version['string_version']}")
        resolved = client.resolve(GENES)
    except RuntimeError as error:
        print(f"  STOPPED: {error}")
        return 2
    missing = [g for g in GENES if g not in resolved]
    if missing:
        print(f"  STOPPED: STRING does not recognise {missing}")
        return 2
    to_query_name = {row["preferredName"].upper(): query for query, row in resolved.items()}
    renamed = {
        q: r["preferredName"] for q, r in resolved.items() if r["preferredName"].upper() != q
    }
    print(
        f"  all {len(GENES)} gene names recognised"
        + (f"; STRING uses other preferred names for {renamed}" if renamed else "")
    )

    banner("Step 4. Network queries and the Tier 1 check")
    results, edge_tables, checks = {}, {}, {}
    for kind in NETWORK_TYPES:
        try:
            rows, raw, address = client.network(GENES, kind, REQUIRED_SCORE)
        except RuntimeError as error:
            print(f"  STOPPED: {error}")
            return 2
        (out_dir / f"STRING_{kind}_raw.json").write_text(raw, encoding="utf-8")
        edges = parse_edges(rows, to_query_name)
        edges.to_csv(out_dir / f"STRING_{kind}_edges.csv", index=False)
        check = check_tier1(tier1, edges)
        results[kind] = {
            "n_edges": len(edges),
            "supported": int(check["in_network"].sum()),
            "with_experiments": int(
                (check["in_network"] & (check["experimental_score"] > 0)).sum()
            ),
            "absent": [(r.gene1, r.gene2) for r in check.itertuples() if not r.in_network],
            "address": address,
        }
        edge_tables[kind], checks[kind] = edges, check
        print(
            f"  {kind:10s} {len(edges):3d} interactions among the {len(GENES)} genes; {results[kind]['supported']} of {len(tier1)} Tier 1 edges present ({results[kind]['with_experiments']} with experimental evidence)"
        )

    combined = tier1[["gene1", "gene2"] + [c for c in ("rho", "padj") if c in tier1.columns]].copy()
    for kind, check in checks.items():
        combined[f"in_{kind}"] = check["in_network"].values
        combined[f"{kind}_score"] = check["combined_score"].values
        combined[f"{kind}_experimental"] = check["experimental_score"].values
    comparison = {}
    if stored_path:
        stored = load_stored(stored_path)
        comparison = {
            kind: compare_with_stored(stored, edge_tables[kind]) for kind in NETWORK_TYPES
        }
        stored_pairs = {(r.gene1, r.gene2) for r in stored.itertuples()}
        combined["in_stored_file"] = [
            (r.gene1, r.gene2) in stored_pairs for r in combined.itertuples()
        ]
    combined.to_csv(out_dir / "Tier1_edge_check.csv", index=False)

    banner("Step 5. Writing the record")
    info = {
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "string_host_queried": client.host,
        "string_version_reported": version["string_version"],
        "stable_address_reported": version["stable_address"],
    }
    parameters = dict(
        info,
        script="string_network_query.py",
        species=SPECIES,
        required_score_0_to_1000=REQUIRED_SCORE,
        required_score_0_to_1=REQUIRED_SCORE / 1000,
        network_types=NETWORK_TYPES,
        genes_queried=GENES,
        gene_names_as_understood_by_string={q: r.get("preferredName") for q, r in resolved.items()},
        caller_identity=CALLER_IDENTITY,
        tier1_rule=TIER1_RULE,
        input_files={
            "correlation_table": str(correlation_path),
            "stored_string_file": str(stored_path) if stored_path else None,
        },
        requests_made={kind: res["address"] for kind, res in results.items()},
        package_versions={
            "python": platform.python_version(),
            "requests": requests.__version__,
            "pandas": pd.__version__,
        },
        results={
            kind: {k: v for k, v in res.items() if k not in ("absent", "address")}
            | {"absent_tier1_pairs": ["-".join(p) for p in res["absent"]]}
            for kind, res in results.items()
        },
    )
    (out_dir / "string_rerun_parameters.json").write_text(
        json.dumps(parameters, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "STRING_rerun_report.md").write_text(
        build_report(info, results, comparison, len(tier1)), encoding="utf-8"
    )
    files = sorted(out_dir.iterdir())
    print(f"  wrote {len(files)} files to {out_dir}")
    if drive is not None and drive.can_write:
        for path in files:
            drive.upload(path, OUTPUT_FOLDER)
        print(f"  uploaded them to the '{OUTPUT_FOLDER}' folder in Google Drive")
    elif drive is not None:
        print("  the Drive token is read-only, so the files were not uploaded")
    print(f"\nRead {out_dir / 'STRING_rerun_report.md'} first.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code and IN_NOTEBOOK:
        print(f"\nThe run stopped with code {exit_code}. Read the message above for the reason.")
    elif exit_code:
        sys.exit(exit_code)
