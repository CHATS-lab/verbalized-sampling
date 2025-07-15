# Read the file
with open('parse_results_for_latex.py', 'r') as f:
    lines = f.readlines()

# Find the line where baseline is printed and add baseline CoT after it
for i, line in enumerate(lines):
    if 'Print baseline' in line and 'baseline CoT' not in line:
        # Find the end of the baseline printing block (next empty line)
        j = i + 1
        while j < len(lines) and lines[j].strip() != '':
            j += 1
        
        # Insert baseline CoT code before the empty line
        baseline_cot_code = [
            '\n',
            '        # Print baseline CoT\n',
            '        baseline_cot = results.get("Baseline CoT")\n',
            '        if baseline_cot and any(v is not None for v in baseline_cot.values()):\n',
            '            print(f"& Baseline CoT & {format_metric(baseline_cot[\'diversity\'], baseline_cot[\'diversity\'] == best_diversity)} & {format_metric(baseline_cot[\'rouge_l\'], baseline_cot[\'rouge_l\'] == best_rouge_l)} & {format_metric(baseline_cot[\'quality\'], baseline_cot[\'quality\'] == best_quality)} \\\\\\\\")\n',
        ]
        
        lines = lines[:j] + baseline_cot_code + lines[j:]
        break

# Write back
with open('parse_results_for_latex.py', 'w') as f:
    f.writelines(lines)

print('File updated successfully')
