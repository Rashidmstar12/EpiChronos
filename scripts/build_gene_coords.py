import urllib.request
import gzip
import io
import json
import os
import sys

def main():
    print("=" * 60)
    print("   EPICHRONOS GENE COORDINATES BUILDER (UCSC RefSeq hg19)")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(base_dir, "epichronos", "resources", "gene_coords.json")
    
    # 1. Check for local file first to bypass network restrictions/timeouts
    local_paths = [
        "refGene.txt.gz",
        "scripts/refGene.txt.gz",
        os.path.join(base_dir, "refGene.txt.gz"),
        os.path.join(base_dir, "scripts", "refGene.txt.gz")
    ]
    
    content = None
    success_source = None
    
    for p in local_paths:
        if os.path.exists(p):
            print(f"Found local RefSeq table file at: {p}")
            try:
                with open(p, 'rb') as local_f:
                    content = local_f.read()
                success_source = f"Local file: {p}"
                print("Loaded local file successfully!")
                break
            except Exception as e:
                print(f"Warning: Failed to read local file {p}. Error: {e}\n")
                
    # 2. If no local file is found, attempt downloads from public mirrors
    if content is None:
        urls = [
            # Highly resilient Amazon S3 Direct REST API mirror (fast, globally available, no regional blocks)
            "https://genome-browser.s3.amazonaws.com/goldenPath/hg19/database/refGene.txt.gz",
            # Primary US server alternative subdomain
            "https://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
            # Primary US server
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
            # Europe mirror alternative path (HTTP)
            "http://hgdownload-euro.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
        ]
        
        for url in urls:
            print(f"Attempting to download RefSeq gene table from:\n  {url}...")
            try:
                req = urllib.request.Request(
                    url, 
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                # Short timeout to fail fast and move to next mirror
                with urllib.request.urlopen(req, timeout=15) as response:
                    content = response.read()
                success_source = url
                print("Download successful!")
                break
            except Exception as e:
                print(f"Warning: Failed to fetch from {url}. Error: {e}\n")
                
    if content is None:
        print("\nError: Could not load RefSeq table.", file=sys.stderr)
        print("Because your local network/proxy is blocking direct connections to UCSC database servers, please:", file=sys.stderr)
        print(f"  1. Download the file manually using your web browser from: https://genome-browser.s3.amazonaws.com/goldenPath/hg19/database/refGene.txt.gz", file=sys.stderr)
        print(f"  2. Place the downloaded 'refGene.txt.gz' file directly inside the folder: {os.path.join(base_dir, 'scripts')}", file=sys.stderr)
        print(f"  3. Re-run this script: python scripts/build_gene_coords.py", file=sys.stderr)
        sys.exit(1)
        
    print("Decompressing and parsing RefSeq data...")
    
    try:
        # Define standard chromosomes
        valid_chroms = {f"chr{i}" for i in range(1, 23)}.union({"chrX", "chrY"})
        
        gene_map = {}
        
        # Decompress in-memory gzipped content
        with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 13:
                    continue
                
                # Column mapping based on refGene schema:
                # index 2: chrom
                # index 4: txStart
                # index 5: txEnd
                # index 12: name2 (gene symbol / identifier)
                chrom = parts[2]
                if chrom not in valid_chroms:
                    continue
                    
                try:
                    tx_start = int(parts[4])
                    tx_end = int(parts[5])
                except ValueError:
                    continue
                    
                gene_symbol = parts[12].strip()
                if not gene_symbol:
                    continue
                
                # Keep coordinates grouped by gene symbol and chrom
                if gene_symbol not in gene_map:
                    gene_map[gene_symbol] = {
                        "chrom": chrom,
                        "starts": [tx_start],
                        "ends": [tx_end]
                    }
                else:
                    # Only group and union range if it's on the same chromosome as first seen
                    if gene_map[gene_symbol]["chrom"] == chrom:
                        gene_map[gene_symbol]["starts"].append(tx_start)
                        gene_map[gene_symbol]["ends"].append(tx_end)
                        
        # Construct final coordinate map with union range (min txStart, max txEnd)
        final_coords = {}
        for gene, data in gene_map.items():
            min_start = min(data["starts"])
            max_end = max(data["ends"])
            final_coords[gene] = [data["chrom"], min_start, max_end]
            
        # Save to JSON
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as out_f:
            json.dump(final_coords, out_f, indent=4)
            
        print(f"\nSuccessfully compiled and wrote {len(final_coords)} RefSeq gene coordinates to:")
        print(f"  {target_path}")
        print(f"Source: {success_source}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError parsing RefSeq table: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
