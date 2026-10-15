import os
import re
import csv
import glob
import sys

def parse_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Try to extract parameters from filename: run_{N}_{M}_{OP}_{DELAY}ms_sec{SEC}.log
    filename = os.path.basename(file_path)
    # run_8_23_1_100ms_sec1.log
    match_file = re.match(r'run_(\d+)_(\d+)_(\d+)_(\d+)ms_sec(\d+)', filename)
    
    data = {
        'Filename': filename,
        'N': match_file.group(1) if match_file else '?',
        'LogSize': match_file.group(2) if match_file else '?',
        'Delay': match_file.group(4) if match_file else '?',
        'Security': match_file.group(5) if match_file else '?',
        'Status': 'Success' if 'OK' in content and 'Vt :' in content else 'Failed',
        'CommitTime': '?',
        'FirstSumcheck': '?',
        'SecondSumcheck': '?',
        'SparseEvalPIOP': '?',
        'BatchWhirProof': '?',
        'VerifyTime': '?',
        'ProofSize': '?',
        'CommMB': '?'
    }

    # Extract metrics using regex
    def get_match(pattern, text, group=1):
        m = re.search(pattern, text)
        return m.group(group) if m else '?'

    data['CommitTime'] = get_match(r'Commit Phase--- Pt: ([\d\.]+)', content)
    data['FirstSumcheck'] = get_match(r'First Sumcheck--- Pt: ([\d\.]+)', content)
    data['SecondSumcheck'] = get_match(r'Second Sumcheck--- Pt: ([\d\.]+)', content)
    data['SparseEvalPIOP'] = get_match(r'Sparse Eval PIOP: ([\d\.]+)', content)
    data['BatchWhirProof'] = get_match(r'BATCH WHIR proof: ([\d\.]+)', content)
    data['VerifyTime'] = get_match(r'Vt\s*:\s*([\d\.]+)\s*sec', content)
    data['ProofSize'] = get_match(r'Ps:\s*([\d\.]+)\s*KB', content)
    data['CommMB'] = get_match(r'Com:\s*([\d\.]+)\s*MB', content)

    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_logs.py <directory_with_logs>")
        return

    log_dir = sys.argv[1]
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    results = []
    for log_file in log_files:
        results.append(parse_log(log_file))

    # Sort by N, then LogSize, then Delay
    results.sort(key=lambda x: (int(x['N']) if x['N'].isdigit() else 0, 
                                int(x['LogSize']) if x['LogSize'].isdigit() else 0, 
                                int(x['Delay']) if x['Delay'].isdigit() else 0))

    output_file = os.path.join(log_dir, "custom_analysis.csv")
    fieldnames = ['N', 'LogSize', 'Delay', 'Security', 'Status', 'CommitTime', 'FirstSumcheck', 'SecondSumcheck', 'SparseEvalPIOP', 'BatchWhirProof', 'VerifyTime', 'ProofSize', 'CommMB', 'Filename']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    main()
