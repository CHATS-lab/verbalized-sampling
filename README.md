# Verbalized Sampling

## Structure

- `scripts/`: All shell scripts for running experiments and batch jobs
- `output/`: Model outputs and response files
- `llms/`: Language model interface code
- `create_plot.py`: Main plotting logic (now module-ready)
- `__main__.py`: Entry point for running as a module

## Usage

### Plotting Histograms

You can run the plotting code as a module:

```bash
python -m verbalized_sampling --target_dir output/meta-llama_Llama-3.3-70B-Instruct --output all_histograms_T1.0_subplots_row.png --sizes 1 3 5 10 50
```

- `--target_dir`: Directory containing the response files (default: output/meta-llama_Llama-3.3-70B-Instruct)
- `--output`: Output image file name (default: all_histograms_T1.0_subplots_row.png)
- `--sizes`: Sampling sizes to plot (default: 1 3 5 10 50)

### Scripts

All `.sh` scripts are now in the `scripts/` directory. Run them as needed, e.g.:

```bash
bash scripts/run.sh
``` 