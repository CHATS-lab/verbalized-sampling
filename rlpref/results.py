import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, binomtest
import statsmodels.stats.proportion as smp



def analyze_axis_results(results, model_name="model"):
    """
    Analyze results from the axis experiment.
    
    Args:
        results: List of result dictionaries from run_axis_experiment
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with analysis results
    """
    if not results:
        print("No results to analyze.")
        return None
    
    # Extract ratings, log probabilities, and token counts
    ratings = [float(r["rating"]) for r in results if r["rating"] is not None]
    avg_logprobs = [float(r["avg_logprob"]) for r in results if r["rating"] is not None]
    sum_logprobs = [float(r["sum_logprob"]) for r in results if r["rating"] is not None]
    token_counts = [int(r["token_count"]) for r in results if r["rating"] is not None and "token_count" in r]
    
    # Verify we have valid numeric data
    if len(ratings) == 0 or len(avg_logprobs) == 0:
        print("Warning: No valid ratings or logprobs found for correlation analysis")
        analysis_results = {
            "model": model_name,
            "num_samples": len(results),
            "pearson_correlation": float('nan'),
            "pearson_p_value": float('nan'),
            "spearman_correlation": float('nan'),
            "spearman_p_value": float('nan')
        }
        return analysis_results
    
    # Make sure we have matching data points
    min_length = min(len(ratings), len(avg_logprobs))
    ratings = ratings[:min_length]
    avg_logprobs = avg_logprobs[:min_length]
    sum_logprobs = sum_logprobs[:min_length]
    
    # Convert to numpy arrays to ensure numeric types
    ratings = np.array(ratings, dtype=float)
    avg_logprobs = np.array(avg_logprobs, dtype=float)
    avg_probs = np.exp(avg_logprobs)  # Convert logprobs to probs for better interpretability
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(ratings, avg_probs)
    spearman_corr, spearman_p = spearmanr(ratings, avg_probs)
    
    # Also calculate correlations for log probs
    log_pearson_corr, log_pearson_p = pearsonr(ratings, avg_logprobs)
    log_spearman_corr, log_spearman_p = spearmanr(ratings, avg_logprobs)
    
    # Get additional metrics if available
    accuracy_corr = coherence_corr = coverage_corr = float('nan')
    try:
        accuracy_ratings = [float(r["accuracy"]) for r in results if r["accuracy"] is not None]
        coherence_ratings = [float(r["coherence"]) for r in results if r["coherence"] is not None]
        coverage_ratings = [float(r["coverage"]) for r in results if r["coverage"] is not None]
        
        # Only calculate if we have data and it matches our data points
        if len(accuracy_ratings) >= min_length:
            accuracy_corr, _ = pearsonr(accuracy_ratings[:min_length], avg_probs[:min_length])
        if len(coherence_ratings) >= min_length:
            coherence_corr, _ = pearsonr(coherence_ratings[:min_length], avg_probs[:min_length])
        if len(coverage_ratings) >= min_length:
            coverage_corr, _ = pearsonr(coverage_ratings[:min_length], avg_probs[:min_length])
    except:
        pass  # If any calculations fail, we already have NaN values
    
    # Create analysis results
    analysis_results = {
        "model": model_name,
        "num_samples": len(results),
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "log_pearson_correlation": log_pearson_corr,
        "log_pearson_p_value": log_pearson_p,
        "log_spearman_correlation": log_spearman_corr,
        "log_spearman_p_value": log_spearman_p,
        "accuracy_correlation": accuracy_corr,
        "coherence_correlation": coherence_corr,
        "coverage_correlation": coverage_corr,
        "data": {
            "ratings": ratings.tolist(),
            "avg_probs": avg_probs.tolist(),
            "avg_logprobs": avg_logprobs.tolist(),
            "token_counts": token_counts.tolist() if hasattr(token_counts, 'tolist') else list(token_counts) if token_counts else []
        }
    }
    
    print("\n=== Axis Analysis Results ===")
    print(f"Number of samples: {len(results)}")
    print(f"Pearson correlation (with probs): {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation (with probs): {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    print(f"Pearson correlation (with logprobs): {log_pearson_corr:.4f} (p-value: {log_pearson_p:.4f})")
    print(f"Spearman correlation (with logprobs): {log_spearman_corr:.4f} (p-value: {log_spearman_p:.4f})")
    
    if not np.isnan(accuracy_corr):
        print("\nCorrelations with different aspects of quality:")
        print(f"  Accuracy: {accuracy_corr:.4f}")
        print(f"  Coherence: {coherence_corr:.4f}")
        print(f"  Coverage: {coverage_corr:.4f}")
    
    return analysis_results


def plot_axis_results(analysis_results, file_suffix="model", results_dir=".", jitter_seed=42, output_format="pdf"):
    """
    Plot the results from the axis experiment analysis.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_axis_results
        file_suffix: Suffix to use in plot filenames (typically model name)
        results_dir: Directory to save the plots
        jitter_seed: Random seed for consistent jittering
        output_format: Format for saving plots ('pdf' or 'png')
        
    Returns:
        Dictionary with paths to saved plots
    """
    # Set random seed for reproducible jitter
    np.random.seed(jitter_seed)
    if not analysis_results:
        print("No analysis results to plot.")
        return {}
    
    plots = {}
    
    # Extract data for plotting
    try:
        ratings = np.array(analysis_results["data"]["ratings"])
        avg_probs = np.array(analysis_results["data"]["avg_probs"])
        avg_logprobs = np.array(analysis_results["data"]["avg_logprobs"])
        pearson_corr = analysis_results["pearson_correlation"]
        pearson_p = analysis_results["pearson_p_value"]
        log_pearson_corr = analysis_results["log_pearson_correlation"]
        log_pearson_p = analysis_results["log_pearson_p_value"]
        
        # Extract token counts if available
        token_counts = []
        for item in analysis_results.get("sample_results", []):
            if "token_count" in item:
                token_counts.append(item["token_count"])
        
        # If token counts aren't in sample_results, try to extract from the main data
        if len(token_counts) == 0 and "token_counts" in analysis_results.get("data", {}):
            token_counts_data = analysis_results["data"]["token_counts"]
            if isinstance(token_counts_data, list) and len(token_counts_data) > 0:
                token_counts = token_counts_data
        
        # If we still don't have token counts, create a dummy array
        if len(token_counts) == 0:
            print("Token count data not found, using default values")
            token_counts = [30] * len(ratings)  # Default token count
        
        # Convert to numpy array
        token_counts = np.array(token_counts)
        
        # Make sure lengths match
        if len(token_counts) != len(ratings):
            print(f"Warning: Mismatch between token counts ({len(token_counts)}) and ratings ({len(ratings)})")
            # Use available token counts for as many points as possible
            max_len = min(len(token_counts), len(ratings))
            token_counts = token_counts[:max_len]
            ratings = ratings[:max_len]
            avg_probs = avg_probs[:max_len]
            avg_logprobs = avg_logprobs[:max_len]
    except KeyError:
        print("Required data fields missing from analysis results.")
        return {}
    
    # 1. Basic scatter plot with correlation (probabilities)
    plt.figure(figsize=(12, 7))  # Slightly wider to accommodate colorbar
    if len(ratings) > 0 and len(avg_probs) > 0:
        # Check if we should use log scale for token counts (if there are outliers)
        min_tokens = np.min(token_counts)
        max_tokens = np.max(token_counts)
        median_tokens = np.median(token_counts)
        
        use_log_scale = max_tokens > 5 * median_tokens
        
        # If we have outliers, use a logarithmic color scale to better distribute colors
        if use_log_scale and min_tokens > 0:
            print(f"Using logarithmic color scale (token count range: {min_tokens} - {max_tokens}, median: {median_tokens})")
            # Add small constant to avoid log(0)
            normalized_counts = np.log1p(token_counts) 
            colorbar_label = 'Token Count (log scale)'
        else:
            print(f"Using linear color scale (token count range: {min_tokens} - {max_tokens})")
            normalized_counts = token_counts
            colorbar_label = 'Token Count'
            
        # Create a custom norm for the colorbar - this ensures smoother color interpolation
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=np.min(normalized_counts), vmax=np.max(normalized_counts))
        
        # Add jitter to ratings for better visualization (only for plotting)
        jittered_ratings = ratings + np.random.uniform(-0.25, 0.25, size=len(ratings))
        
        # Create scatter plot with token count as color
        scatter = plt.scatter(avg_probs, jittered_ratings, 
                             c=normalized_counts, alpha=0.7, 
                             cmap='viridis', s=70,  # Size of points
                             edgecolors='black', linewidths=0.5,  # Add edge for visibility
                             norm=norm)
        
        # Add a colorbar with custom ticks if using log scale
        cbar = plt.colorbar(scatter)
        cbar.set_label(colorbar_label)
        
        # If we're using log scale, add custom tick labels showing actual token counts
        if use_log_scale and len(token_counts) > 1:
            try:
                # Generate tick positions spanning the range of log-transformed values
                import matplotlib.ticker as ticker
                
                # Create 5 tick positions from min to max
                tick_positions = np.linspace(
                    np.min(normalized_counts), 
                    np.max(normalized_counts), 
                    num=5
                )
                
                # Convert back from log space to original token counts
                tick_labels = [f"{int(np.expm1(pos))}" for pos in tick_positions]
                
                # Set the tick positions and labels
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
            except Exception as e:
                print(f"Could not create custom tick labels: {e}")
        
        # Only add trendline if we have valid correlation
        if not np.isnan(pearson_corr):
            # Add a trend line (using original non-jittered ratings for the calculation)
            try:
                # Calculate trendline using original data
                z = np.polyfit(avg_probs, ratings, 1)
                p = np.poly1d(z)
                
                # Sort x values for smooth line plotting
                sorted_indices = np.argsort(avg_probs)
                sorted_x = avg_probs[sorted_indices]
                
                # Plot the trend line
                plt.plot(sorted_x, p(sorted_x), "r--", alpha=0.7, linewidth=2)
            except Exception as e:
                print(f"Could not plot trend line: {e}")
        
        plt.xlabel("Average Token Probability")
        plt.ylabel("Human Rating (1-10)")
        
        # Display correlation in title if valid
        if np.isnan(pearson_corr):
            plt.title(f"Human Rating vs. Model Token Probability\n(Insufficient data for correlation)")
        else:
            plt.title(f"Human Rating vs. Model Token Probability\nPearson r: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
    else:
        plt.text(0.5, 0.5, "No valid data points to plot", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    # Save the plot to the results directory
    plot_filename = os.path.join(results_dir, f"axis_prob_{file_suffix}.{output_format}")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plots["probability_scatter"] = plot_filename
    
    # 2. Scatter plot with logprobs
    plt.figure(figsize=(12, 7))  # Slightly wider to accommodate colorbar
    if len(ratings) > 0 and len(avg_logprobs) > 0:
        # Check if we should use log scale for token counts (if there are outliers)
        min_tokens = np.min(token_counts)
        max_tokens = np.max(token_counts)
        median_tokens = np.median(token_counts)
        
        use_log_scale = max_tokens > 5 * median_tokens
        
        # If we have outliers, use a logarithmic color scale to better distribute colors
        if use_log_scale and min_tokens > 0:
            # Add small constant to avoid log(0)
            normalized_counts = np.log1p(token_counts) 
            colorbar_label = 'Token Count (log scale)'
        else:
            normalized_counts = token_counts
            colorbar_label = 'Token Count'
            
        # Create a custom norm for the colorbar 
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=np.min(normalized_counts), vmax=np.max(normalized_counts))
        
        # Add jitter to ratings for better visualization (only for plotting)
        jittered_ratings = ratings + np.random.uniform(-0.25, 0.25, size=len(ratings))
        
        # Create scatter plot with token count as color
        scatter = plt.scatter(avg_logprobs, jittered_ratings, 
                             c=normalized_counts, alpha=0.7, 
                             cmap='viridis', s=70,  # Size of points
                             edgecolors='black', linewidths=0.5,  # Add edge for visibility
                             norm=norm)
        
        # Add a colorbar with custom ticks if using log scale
        cbar = plt.colorbar(scatter)
        cbar.set_label(colorbar_label)
        
        # If we're using log scale, add custom tick labels
        if use_log_scale and len(token_counts) > 1:
            try:
                # Generate tick positions spanning the range of log-transformed values
                # Create 5 tick positions from min to max
                tick_positions = np.linspace(
                    np.min(normalized_counts), 
                    np.max(normalized_counts), 
                    num=5
                )
                
                # Convert back from log space to original token counts
                tick_labels = [f"{int(np.expm1(pos))}" for pos in tick_positions]
                
                # Set the tick positions and labels
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
            except Exception as e:
                print(f"Could not create custom tick labels for logprob plot: {e}")
        
        # Only add trendline if we have valid correlation
        if not np.isnan(log_pearson_corr):
            # Add a trend line (using original non-jittered ratings for the calculation)
            try:
                # Calculate trendline using original data
                z = np.polyfit(avg_logprobs, ratings, 1)
                p = np.poly1d(z)
                
                # Sort x values for smooth line plotting
                sorted_indices = np.argsort(avg_logprobs)
                sorted_x = avg_logprobs[sorted_indices]
                
                # Plot the trend line
                plt.plot(sorted_x, p(sorted_x), "r--", alpha=0.7, linewidth=2)
            except Exception as e:
                print(f"Could not plot trend line: {e}")
        
        plt.xlabel("Average Log Probability")
        plt.ylabel("Human Rating (1-10)")
        
        # Display correlation in title if valid
        if np.isnan(log_pearson_corr):
            plt.title(f"Human Rating vs. Model Log Probability\n(Insufficient data for correlation)")
        else:
            plt.title(f"Human Rating vs. Model Log Probability\nPearson r: {log_pearson_corr:.4f}, p-value: {log_pearson_p:.4f}")
    else:
        plt.text(0.5, 0.5, "No valid data points to plot", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    # Save the plot to the results directory
    plot_filename = os.path.join(results_dir, f"axis_logprob_{file_suffix}.{output_format}")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plots["logprob_scatter"] = plot_filename
    
    # 3. If we have a lot of data points, create a hexbin plot for better visualization
    if len(ratings) > 20:
        plt.figure(figsize=(12, 7))
        # Create hexbin plot
        hb = plt.hexbin(avg_probs, ratings, gridsize=15, cmap="plasma", mincnt=1)
        # Add colorbar with better styling
        cbar = plt.colorbar(hb, label="Data Points per Hexbin")
        cbar.ax.tick_params(labelsize=9)
        
        plt.xlabel("Average Token Probability")
        plt.ylabel("Human Rating (1-10)")
        
        # Add grid and improved styling
        plt.grid(alpha=0.3)
        plt.title(f"Human Rating vs. Model Token Probability (Hexbin)\nPearson r: {pearson_corr:.4f}")
        
        # Save the hexbin plot
        hexbin_filename = os.path.join(results_dir, f"axis_hexbin_{file_suffix}.{output_format}")
        plt.tight_layout()
        plt.savefig(hexbin_filename)
        plots["hexbin"] = hexbin_filename
    
    # 4. Plot correlations for different metrics if available
    if all(k in analysis_results for k in ["accuracy_correlation", "coherence_correlation", "coverage_correlation"]):
        # Bar chart of correlations for different metrics
        plt.figure(figsize=(10, 6))
        metrics = ["Overall", "Accuracy", "Coherence", "Coverage"]
        correlations = [
            analysis_results["pearson_correlation"],
            analysis_results["accuracy_correlation"],
            analysis_results["coherence_correlation"],
            analysis_results["coverage_correlation"]
        ]
        
        colors = ["blue" if c >= 0 else "red" for c in correlations]
        plt.bar(metrics, correlations, color=colors)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylim(-1, 1)
        plt.ylabel("Pearson Correlation with Token Probability")
        plt.title("Correlation between Model Probability and Human Ratings")
        
        # Save the metrics correlation plot
        metrics_filename = os.path.join(results_dir, f"axis_metrics_{file_suffix}.{output_format}")
        plt.tight_layout()
        plt.savefig(metrics_filename)
        plots["metrics_comparison"] = metrics_filename
    
    # 5. Special bimodal analysis plot - split by probability threshold with flipped axes
    plt.figure(figsize=(12, 8))
    if len(ratings) > 0 and len(avg_probs) > 0:
        # Define the threshold for splitting the data
        PROB_THRESHOLD = 0.7
        
        # Split the data into two groups based on probability
        low_mask = avg_probs < PROB_THRESHOLD
        high_mask = avg_probs >= PROB_THRESHOLD
        
        # Get the data points for each group
        low_probs = avg_probs[low_mask]
        low_ratings = ratings[low_mask]
        high_probs = avg_probs[high_mask]
        high_ratings = ratings[high_mask]
        
        # Count points in each group
        low_count = np.sum(low_mask)
        high_count = np.sum(high_mask)
        
        # Create jittered ratings for better visualization (without changing the original for line fitting)
        low_jittered = low_ratings + np.random.uniform(-0.25, 0.25, size=len(low_ratings))
        high_jittered = high_ratings + np.random.uniform(-0.25, 0.25, size=len(high_ratings))
        
        # Plot low probability group (probability < threshold)
        if len(low_probs) > 0:
            # Calculate trend line for low probabilities if we have enough points
            # With flipped axes, we're predicting probability from rating
            if len(low_ratings) > 1:
                try:
                    # Note the flipped inputs to polyfit compared to before - ratings is now X
                    z_low = np.polyfit(low_ratings, low_probs, 1)
                    p_low = np.poly1d(z_low)
                    
                    # Calculate correlation for this subset (order doesn't matter for Pearson r)
                    low_corr, low_p = pearsonr(low_ratings, low_probs)
                    
                    # Create x values for plotting the line (within this range only)
                    # For flipped axes, x is now ratings
                    x_low = np.linspace(min(low_ratings), max(low_ratings), 100)
                    
                    # Only plot trend line if correlation is significant
                    plt.plot(x_low, p_low(x_low), 'b--', 
                           linewidth=2)
                           # label=f'Low prob trend (r={low_corr:.2f}, p={low_p:.3f})')

                except Exception as e:
                    print(f"Could not calculate trend line for low probabilities: {e}")
        
                # Create scatter plot for low probs - note the flipped axes: X=ratings, Y=probs
                plt.scatter(low_jittered, low_probs, 
                        alpha=0.7, color='blue', label=f'Primary cluster: Prob<{PROB_THRESHOLD}, n={low_count}, r={low_corr:.2f} (p={low_p:.3f})')
                
        # Plot high probability group (probability >= threshold)
        if len(high_probs) > 0:
            # Calculate trend line for high probabilities if we have enough points
            if len(high_ratings) > 1:
                try:
                    z_high = np.polyfit(high_ratings, high_probs, 1)
                    p_high = np.poly1d(z_high)
                    
                    # Calculate correlation for this subset
                    high_corr, high_p = pearsonr(high_ratings, high_probs)
                    
                    # Create x values for plotting the line (within this range only)
                    x_high = np.linspace(min(high_ratings), max(high_ratings), 100)
                    
                    # Only plot trend line if correlation is significant
                    plt.plot(x_high, p_high(x_high), '--', 
                           linewidth=2, 
                           # label=f'High prob trend (r={high_corr:.2f}, p={high_p:.3f})',
                           color='gray')
                except Exception as e:
                    print(f"Could not calculate trend line for high probabilities: {e}")
        
            # Create scatter plot for high probs - with flipped axes
            plt.scatter(high_jittered, high_probs, 
                       alpha=0.7, color='gray', label=f'Suspected memorization: Prob≥{PROB_THRESHOLD}, n={high_count}, r={high_corr:.2f} (p={high_p:.3f})')

        # Add a horizontal line at the threshold (horizontal since we flipped axes)
        plt.axhline(y=PROB_THRESHOLD, color='k', linestyle='-', alpha=0.5) # , label=f'Cluster Cutoff: {PROB_THRESHOLD}')
        
        # Add overall correlation info
        overall_corr, overall_p = pearsonr(ratings, avg_probs)  # Flipped order but r is the same
        # plt.title(f"Bimodal Analysis: Token Probability (Y) vs. Human Rating (X)\nOverall correlation: r={overall_corr:.2f}, p={overall_p:.3f}")
        
        # Calculate percentage in each group
        pct_low = (low_count / len(avg_probs)) * 100
        pct_high = (high_count / len(avg_probs)) * 100
        
        # Add distribution info in a text box
        # info_text = (f"Distribution:\n"
        #             f"  Low prob (<{PROB_THRESHOLD}): {pct_low:.1f}% ({low_count} points)\n"
        #             f"  High prob (≥{PROB_THRESHOLD}): {pct_high:.1f}% ({high_count} points)")
        # plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction', 
        #            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        #            fontsize=10)
        
        # Flipped labels
        plt.ylabel("Average Token Probabilities")
        plt.xlabel("Human Rating: Likert scale 1-7 (jitter added for visibility)")
        
        # Set plot limits for better visualization
        plt.xlim(0.5, 7.5)  # Add some padding for ratings 1-10
        plt.ylim(0, 1)  # Probabilities are 0-1 with small padding
        
        plt.legend(loc='center left', bbox_to_anchor=(0.05, -0.10), ncol=2)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No valid data points to plot", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
    
    # Save the plot
    bimodal_filename = os.path.join(results_dir, f"axis_bimodal_{file_suffix}.{output_format}")
    plt.tight_layout()
    plt.savefig(bimodal_filename)
    plots["bimodal_analysis"] = bimodal_filename
    
    return plots


def analyze_comparisons_results(results, model_name="model"):
    """
    Analyze results from the comparisons experiment.
    
    Args:
        results: List of result dictionaries from run_comparisons_experiment
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with analysis results
    """
    if not results:
        print("No results to analyze.")
        return None
    
    # Calculate aggregate statistics
    total_examples = len(results)
    agrees_count = sum(1 for r in results if r["model_preferred_chosen"])
    model_chosen_agreement = agrees_count / total_examples if total_examples > 0 else 0
    
    # Create a 2x2 grid
    # Higher logprob - chosen, Higher logprob - not chosen
    grid = {
        "higher_logprob_chosen": agrees_count,
        "higher_logprob_rejected": total_examples - agrees_count,
        "total": total_examples
    }
    
    # Calculate p-value for the agreement ratio using binomial test
    # Null hypothesis: Agreement is by chance (coin flip probability = 0.5)
    # We use the binomial test with p=0.5 (null hypothesis is random guessing)
    binom_result = binomtest(agrees_count, n=total_examples, p=0.5)
    p_value = binom_result.pvalue
    
    # Calculate confidence interval for the agreement probability (Wilson score interval)
    conf_interval = smp.proportion_confint(agrees_count, total_examples, alpha=0.05, method='wilson')
    
    # Record all these statistics
    agreement_stats = {
        "agreement_ratio": model_chosen_agreement,
        "agrees_count": agrees_count,
        "total_count": total_examples,
        "p_value": p_value,
        "significant": p_value < 0.05,  # Standard threshold
        "conf_interval_lower": float(conf_interval[0]),
        "conf_interval_upper": float(conf_interval[1])
    }
    
    # Calculate log prob differences
    log_prob_diffs = []
    prob_ratios = []
    for result in results:
        chosen_logprob = result["chosen_avg_logprob"]
        rejected_logprob = result["rejected_avg_logprob"]
        log_prob_diffs.append(chosen_logprob - rejected_logprob)
        
        # Calculate probability ratio (chosen/rejected)
        chosen_prob = np.exp(chosen_logprob)
        rejected_prob = np.exp(rejected_logprob)
        if rejected_prob > 0:
            prob_ratios.append(chosen_prob / rejected_prob)
        else:
            prob_ratios.append(float('inf'))  # Infinite ratio if rejected prob is 0
    
    # Additional statistics about the distribution of log prob differences
    log_prob_diffs = np.array(log_prob_diffs)
    prob_ratios = np.array([ratio for ratio in prob_ratios if not np.isinf(ratio)])  # Filter out infinities
    
    # Create analysis results
    analysis_results = {
        "model": model_name,
        "num_samples": total_examples,
        "model_human_agreement": model_chosen_agreement,
        "grid": grid,
        "agreement_stats": agreement_stats,
        "data": {
            "log_prob_diffs": log_prob_diffs.tolist(),
            "prob_ratios": prob_ratios.tolist()
        }
    }
    
    # Add statistics about the distribution of differences if we have data
    if len(log_prob_diffs) > 0:
        analysis_results.update({
            "mean_log_prob_diff": float(np.mean(log_prob_diffs)),
            "median_log_prob_diff": float(np.median(log_prob_diffs)),
            "std_log_prob_diff": float(np.std(log_prob_diffs)),
            "min_log_prob_diff": float(np.min(log_prob_diffs)),
            "max_log_prob_diff": float(np.max(log_prob_diffs))
        })
    
    if len(prob_ratios) > 0:
        analysis_results.update({
            "mean_prob_ratio": float(np.mean(prob_ratios)),
            "median_prob_ratio": float(np.median(prob_ratios)),
            "std_prob_ratio": float(np.std(prob_ratios))
        })
    
    print("\n=== Comparisons Analysis Results ===")
    print(f"Number of samples: {total_examples}")
    print(f"Model-human agreement: {model_chosen_agreement:.2%} ({agrees_count}/{total_examples})")
    
    # Report statistical significance
    significance_marker = "*" if p_value < 0.05 else ""
    print(f"Statistical significance: p-value = {p_value:.4f}{significance_marker}")
    print(f"95% confidence interval: [{conf_interval[0]:.2%}, {conf_interval[1]:.2%}]")
    
    if len(log_prob_diffs) > 0:
        print("\nLog probability difference statistics (chosen - rejected):")
        print(f"  Mean: {analysis_results['mean_log_prob_diff']:.4f}")
        print(f"  Median: {analysis_results['median_log_prob_diff']:.4f}")
        print(f"  Standard deviation: {analysis_results['std_log_prob_diff']:.4f}")
    
    return analysis_results


def plot_comparisons_results(analysis_results, file_suffix="model", results_dir=".", jitter_seed=42, output_format="pdf"):
    """
    Plot the results from the comparisons experiment analysis.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_comparisons_results
        file_suffix: Suffix to use in plot filenames (typically model name)
        results_dir: Directory to save the plots
        jitter_seed: Random seed for consistent jittering
        output_format: Format for saving plots ('pdf' or 'png')
        
    Returns:
        Dictionary with paths to saved plots
    """
    # Set random seed for reproducible jitter
    np.random.seed(jitter_seed)
    if not analysis_results:
        print("No analysis results to plot.")
        return {}
    
    plots = {}
    
    # 1. Basic agreement bar chart
    plt.figure(figsize=(8, 8))
    grid = analysis_results.get("grid", {})
    if grid and grid.get("total", 0) > 0:
        # Bar chart of model agreement with human preferences
        agrees = grid.get("higher_logprob_chosen", 0)
        disagrees = grid.get("higher_logprob_rejected", 0)
        total = grid.get("total", 0)
        
        # Use Seaborn's color palette
        colors = sns.color_palette("Set2")
        
        plt.bar(
            ["Logprobs aligned with annotator preferences", "Logprobs misaligned with annotator preferences"],
            [agrees, disagrees],
            color=colors,
        )
        
        # Get agreement statistics
        if "agreement_stats" in analysis_results:
            stats = analysis_results["agreement_stats"]
            p_value = stats.get("p_value", 1.0)
            conf_low = stats.get("conf_interval_lower", 0.0)
            conf_high = stats.get("conf_interval_upper", 1.0)
            is_significant = p_value < 0.05
            
            # Add significance marker to title if appropriate
            sig_marker = "*" if is_significant else ""
            agreement_pct = analysis_results['model_human_agreement']

            title = f"Model-Human Agreement: {agreement_pct:.2%}{sig_marker}"
            
            # Add p-value and confidence interval to title
            title += f"\np-value: {p_value:.4f}, 95% CI: [{conf_low:.2%}, {conf_high:.2%}]"
            
            # Add a note about statistical significance
            if is_significant:
                title += "\n(Statistically significant)"
            else:
                title += "\n(Not statistically significant)"
                
            #plt.title(title)
        #else:
            # Fallback if agreement stats not available
            #plt.title(f"Model-Human Agreement: {analysis_results['model_human_agreement']:.2%}")
        
        plt.ylabel("Number of examples")
        
        # Add percentages to the bars
        for i, count in enumerate([agrees, disagrees]):
            plt.text(i, count + 0.5, f"{count/total:.1%}", 
                     horizontalalignment='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No valid data points to plot", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    # Save the plot to the results directory
    bar_filename = os.path.join(results_dir, f"comparisons_agreement_{file_suffix}.{output_format}")
    plt.tight_layout()
    plt.savefig(bar_filename)
    plots["agreement_bar"] = bar_filename
    
    # 2. Histogram of log probability differences
    if "data" in analysis_results and "log_prob_diffs" in analysis_results["data"]:
        log_prob_diffs = np.array(analysis_results["data"]["log_prob_diffs"])
        mean_log_prob_diff = np.mean(log_prob_diffs)
        if len(log_prob_diffs) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(log_prob_diffs, bins=20, alpha=0.7, color="blue")
            plt.axvline(x=mean_log_prob_diff, color='red', linestyle='--', alpha=0.7)
            plt.xlabel("Log Probability Difference (Chosen - Rejected)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Log Probability Differences")
            
            # Save the histogram plot
            hist_filename = os.path.join(results_dir, f"comparisons_logdiff_{file_suffix}.{output_format}")
            plt.tight_layout()
            plt.savefig(hist_filename)
            plots["logdiff_histogram"] = hist_filename
    
    # 3. Histogram of probability ratios (if available and not all infinite)
    if "data" in analysis_results and "prob_ratios" in analysis_results["data"]:
        prob_ratios = np.array(analysis_results["data"]["prob_ratios"])
        if len(prob_ratios) > 5:  # Only if we have meaningful data
            plt.figure(figsize=(10, 6))
            mean_prob_ratio = np.mean(prob_ratios)
            # Use log scale for better visibility if ranges are large
            if np.max(prob_ratios) / np.median(prob_ratios) > 10:
                plt.hist(np.log10(prob_ratios), bins=20, alpha=0.7, color="green")
                plt.xlabel("Log10 of Probability Ratio (Chosen/Rejected)")
                plt.axvline(x=np.log10(mean_prob_ratio), color='red', linestyle='--', alpha=0.7)
                title = "Distribution of Log10 Probability Ratios"
            else:
                plt.hist(prob_ratios, bins=20, alpha=0.7, color="green")
                plt.xlabel("Probability Ratio (Chosen/Rejected)")
                plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                title = "Distribution of Probability Ratios"
            
            plt.ylabel("Frequency")
            plt.title(title)
            
            # Save the ratio histogram plot
            ratio_filename = os.path.join(results_dir, f"comparisons_ratio_{file_suffix}.{output_format}")
            plt.tight_layout()
            plt.savefig(ratio_filename)
            plots["ratio_histogram"] = ratio_filename
    
    return plots
