#!/usr/bin/env python3
"""
Script to count the number of parts in each object from preprocessed data.
Generates statistics and visualizations of part count distribution.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def count_parts_from_config(config_path):
    """
    Count parts from object_part_configs.json file.

    Args:
        config_path (str): Path to object_part_configs.json

    Returns:
        dict: Dictionary with statistics and part counts
    """
    with open(config_path, 'r') as f:
        configs = json.load(f)

    # Extract part counts
    part_counts = []
    valid_objects = []
    invalid_objects = []

    for config in configs:
        if config.get('valid', False):
            num_parts = config.get('num_parts', 0)
            part_counts.append(num_parts)
            valid_objects.append({
                'file': config.get('file', 'unknown'),
                'num_parts': num_parts
            })
        else:
            invalid_objects.append(config.get('file', 'unknown'))

    return {
        'part_counts': part_counts,
        'valid_objects': valid_objects,
        'invalid_objects': invalid_objects
    }


def print_statistics(data):
    """Print statistics about part counts."""
    part_counts = data['part_counts']
    valid_objects = data['valid_objects']
    invalid_objects = data['invalid_objects']

    if not part_counts:
        print("No valid objects found!")
        return

    print("\n" + "="*60)
    print("PART COUNT STATISTICS")
    print("="*60)

    print(f"\nTotal objects: {len(valid_objects) + len(invalid_objects)}")
    print(f"Valid objects: {len(valid_objects)}")
    print(f"Invalid objects: {len(invalid_objects)}")

    print(f"\n--- Part Count Statistics ---")
    print(f"Total parts: {sum(part_counts)}")
    print(f"Mean parts per object: {np.mean(part_counts):.2f}")
    print(f"Median parts per object: {np.median(part_counts):.2f}")
    print(f"Std dev: {np.std(part_counts):.2f}")
    print(f"Min parts: {min(part_counts)}")
    print(f"Max parts: {max(part_counts)}")

    # Distribution breakdown
    print(f"\n--- Part Count Distribution ---")
    counter = Counter(part_counts)
    for num_parts in sorted(counter.keys()):
        count = counter[num_parts]
        percentage = (count / len(part_counts)) * 100
        print(f"{num_parts:3d} parts: {count:4d} objects ({percentage:5.2f}%)")

    # Range distribution
    print(f"\n--- Range Distribution ---")
    ranges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    for min_val, max_val in ranges:
        if max_val == float('inf'):
            count = sum(1 for p in part_counts if p >= min_val)
            range_str = f"{min_val}+ parts"
        else:
            count = sum(1 for p in part_counts if min_val <= p <= max_val)
            range_str = f"{min_val}-{max_val} parts"
        percentage = (count / len(part_counts)) * 100
        print(f"{range_str:15s}: {count:4d} objects ({percentage:5.2f}%)")

    # Top 10 objects with most parts
    print(f"\n--- Top 10 Objects with Most Parts ---")
    sorted_objects = sorted(valid_objects, key=lambda x: x['num_parts'], reverse=True)
    for i, obj in enumerate(sorted_objects[:10], 1):
        print(f"{i:2d}. {obj['file']:20s}: {obj['num_parts']:3d} parts")

    print("\n" + "="*60)


def plot_distribution(part_counts, output_path):
    """
    Create visualization plots for part count distribution.

    Args:
        part_counts (list): List of part counts
        output_path (str): Path to save the plot
    """
    if not part_counts:
        print("No data to plot!")
        return

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Part Count Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram
    ax1 = axes[0, 0]
    max_parts = max(part_counts)
    bins = min(50, max_parts)  # Use at most 50 bins
    ax1.hist(part_counts, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Number of Parts', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram of Part Counts', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(part_counts):.2f}\n'
    stats_text += f'Median: {np.median(part_counts):.2f}\n'
    stats_text += f'Std: {np.std(part_counts):.2f}\n'
    stats_text += f'Min: {min(part_counts)}\n'
    stats_text += f'Max: {max(part_counts)}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, family='monospace')

    # 2. Bar chart of part count frequency (for smaller counts)
    ax2 = axes[0, 1]
    counter = Counter(part_counts)
    # Show only part counts that appear at least once, up to 20
    max_display = min(20, max(counter.keys()))
    x_values = list(range(1, max_display + 1))
    y_values = [counter.get(x, 0) for x in x_values]

    bars = ax2.bar(x_values, y_values, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Number of Parts', fontsize=12)
    ax2.set_ylabel('Number of Objects', fontsize=12)
    ax2.set_title(f'Part Count Frequency (1-{max_display} parts)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x_values)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_counts = np.sort(part_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax3.plot(sorted_counts, cumulative, linewidth=2, color='green')
    ax3.set_xlabel('Number of Parts', fontsize=12)
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax3.set_title('Cumulative Distribution of Part Counts', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(0, 100)

    # Add percentile lines
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(part_counts, p)
        ax3.axvline(val, color='red', linestyle='--', alpha=0.5)
        ax3.text(val, 5, f'P{p}={val:.0f}', rotation=90,
                verticalalignment='bottom', fontsize=9)

    # 4. Range distribution (pie chart)
    ax4 = axes[1, 1]
    ranges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    range_labels = []
    range_counts = []

    for min_val, max_val in ranges:
        if max_val == float('inf'):
            count = sum(1 for p in part_counts if p >= min_val)
            label = f'{min_val}+ parts'
        else:
            count = sum(1 for p in part_counts if min_val <= p <= max_val)
            if min_val == max_val:
                label = f'{min_val} part' if min_val == 1 else f'{min_val} parts'
            else:
                label = f'{min_val}-{max_val} parts'

        if count > 0:  # Only include non-empty ranges
            range_labels.append(label)
            range_counts.append(count)

    colors = plt.cm.Set3(np.linspace(0, 1, len(range_counts)))
    wedges, texts, autotexts = ax4.pie(range_counts, labels=range_labels, autopct='%1.1f%%',
                                         startangle=90, colors=colors)
    ax4.set_title('Part Count Range Distribution', fontsize=14, fontweight='bold')

    # Improve text readability
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF
    output_pdf = output_path.with_suffix('.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")

    plt.close()


def save_detailed_report(data, output_path):
    """Save detailed report to a text file."""
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DETAILED PART COUNT REPORT\n")
        f.write("="*60 + "\n\n")

        part_counts = data['part_counts']
        valid_objects = data['valid_objects']
        invalid_objects = data['invalid_objects']

        f.write(f"Total objects: {len(valid_objects) + len(invalid_objects)}\n")
        f.write(f"Valid objects: {len(valid_objects)}\n")
        f.write(f"Invalid objects: {len(invalid_objects)}\n\n")

        if part_counts:
            f.write("--- Statistics ---\n")
            f.write(f"Total parts: {sum(part_counts)}\n")
            f.write(f"Mean: {np.mean(part_counts):.2f}\n")
            f.write(f"Median: {np.median(part_counts):.2f}\n")
            f.write(f"Std dev: {np.std(part_counts):.2f}\n")
            f.write(f"Min: {min(part_counts)}\n")
            f.write(f"Max: {max(part_counts)}\n\n")

            f.write("--- All Valid Objects ---\n")
            sorted_objects = sorted(valid_objects, key=lambda x: x['num_parts'], reverse=True)
            for obj in sorted_objects:
                f.write(f"{obj['file']:30s}: {obj['num_parts']:3d} parts\n")

        if invalid_objects:
            f.write("\n--- Invalid Objects ---\n")
            for obj in invalid_objects:
                f.write(f"{obj}\n")

    print(f"Detailed report saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count and visualize part distribution from object_part_configs.json'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to object_part_configs.json file')
    parser.add_argument('--output', type=str, default='part_count_distribution.png',
                        help='Output path for the plot (default: part_count_distribution.png)')
    parser.add_argument('--report', type=str, default='part_count_report.txt',
                        help='Output path for detailed report (default: part_count_report.txt)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    print(f"Reading config file: {args.input}")

    # Count parts
    data = count_parts_from_config(args.input)

    # Print statistics
    print_statistics(data)

    # Generate plots
    if not args.no_plot and data['part_counts']:
        print("\nGenerating visualization plots...")
        try:
            plot_distribution(data['part_counts'], args.output)
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")

    # Save detailed report
    if data['part_counts']:
        save_detailed_report(data, args.report)

    print("\nDone!")
