#!/usr/bin/env python3
"""Streamlit app for visualizing learning patterns."""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import re
import html
from qhchina.helpers import load_fonts
load_fonts()

# Configure page
st.set_page_config(
    page_title="Pattern Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_pattern_analysis(uploaded_file):
    try:
        content = json.loads(uploaded_file.getvalue().decode("utf-8"))
        
        # Validate JSON structure
        required_fields = ["experiment", "statistics", "patterns"]
        if not all(field in content for field in required_fields):
            return None, "Invalid JSON format: missing required fields"
        
        # Check if it's from find_learning_patterns
        if content.get("experiment") != "exp2_find_learning_patterns":
            return None, "This file does not appear to be from find_significant_patterns.py (updated version)"
        
        return content, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, f"Error loading file: {e}"

def get_pattern_with_context(pattern: Dict, left_context: int, right_context: int) -> str:
    start_idx = pattern['start_idx']
    end_idx = pattern['end_idx']
    full_text = pattern['full_document_text']
    
    # Calculate context boundaries
    context_start = max(0, start_idx - left_context)
    context_end = min(len(full_text), end_idx + right_context)
    
    # Extract text with context
    before_pattern = full_text[context_start:start_idx]
    pattern_text = full_text[start_idx:end_idx]
    after_pattern = full_text[end_idx:context_end]
    
    return before_pattern, pattern_text, after_pattern

def is_pattern_visible(pattern: Dict, filter_chars: str) -> bool:
    if not filter_chars:
        return True  # No filter, show all patterns
    
    pattern_text = pattern['pattern_text']
    
    # Hide pattern if it contains any of the filter characters
    for char in filter_chars:
        if char in pattern_text:
            return False
    
    return True

def extract_visualization_data(pattern: Dict, viz_start: int, viz_end: int) -> Tuple[Dict, Optional[str]]:
    try:
        # Get available epochs
        available_epochs = list(pattern['full_document_epoch_data'].keys())
        available_epochs.sort(key=lambda x: int(x))
        
        # Extract perplexities for each epoch in the visualization range
        epoch_perplexities = {}
        for epoch in available_epochs:
            epoch_data = pattern['full_document_epoch_data'][epoch]
            
            # Extract perplexities from token data tuples (token, token_id, perplexity)
            viz_perplexities = []
            for i in range(viz_start, viz_end):
                if i < len(epoch_data):
                    _, _, perplexity = epoch_data[i]
                    viz_perplexities.append(perplexity)
                else:
                    return None, f"Visualization range {viz_start}:{viz_end} exceeds document length"
            
            epoch_perplexities[int(epoch)] = viz_perplexities
        
        # Get tokens for the visualization range
        full_tokens = pattern['full_document_tokens']
        viz_tokens = full_tokens[viz_start:viz_end]
        
        return {
            'type': 'multi_epoch',
            'epoch_perplexities': epoch_perplexities,
            'epoch_numbers': [int(e) for e in available_epochs],
            'tokens': viz_tokens,
            'perplexity_positions': list(range(viz_start, viz_end))
        }, None
        
    except Exception as e:
        return None, f"Error extracting visualization data: {str(e)}"

def plot_perplexity_distribution(pattern_data: Dict, use_log_scale: bool = True, 
                                alpha: float = 0.7, start_color: str = "#00FFBD", end_color: str = "#002BFF",
                                width: int = 800, height: int = 500, marker_size: int = 4, 
                                legend_position: str = "upper right", line_width: int = 2,
                                show_grid: bool = True, grid_alpha: float = 0.3, font_size: int = 12,
                                bg_color: str = "White"):
    
    # Helper functions
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def generate_color_gradient(n_colors, start_hex, end_hex):
        if n_colors <= 1:
            return [start_hex]
        
        start_r, start_g, start_b = hex_to_rgb(start_hex)
        end_r, end_g, end_b = hex_to_rgb(end_hex)
        
        colors = []
        for i in range(n_colors):
            ratio = i / (n_colors - 1)
            r = int(start_r + (end_r - start_r) * ratio)
            g = int(start_g + (end_g - start_g) * ratio)
            b = int(start_b + (end_b - start_b) * ratio)
            colors.append(f'rgb({r}, {g}, {b})')
        return colors
    
    def format_epoch_name(epoch_num):
        if epoch_num == -1:
            return "Base"
        else:
            return f"Epoch {epoch_num + 1}"
    
    fig = go.Figure()
    
    # Determine data length and create x_positions
    data_length = 0
    if pattern_data.get('type') == 'multi_epoch':
        # Multi-epoch visualization
        epoch_perps = pattern_data.get('epoch_perplexities', {})
        epoch_numbers = pattern_data.get('epoch_numbers', [])
        
        if epoch_perps and epoch_numbers:
            # Get data length from first available epoch
            first_epoch = epoch_numbers[0]
            if first_epoch in epoch_perps:
                data_length = len(epoch_perps[first_epoch])
        
        colors = generate_color_gradient(len(epoch_numbers), start_color, end_color)
        
        for i, epoch_num in enumerate(epoch_numbers):
            if epoch_num in epoch_perps:
                perplexities = epoch_perps[epoch_num]
                x_positions = list(range(len(perplexities)))
                
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=perplexities,
                    mode='lines+markers',
                    name=format_epoch_name(epoch_num),
                    line=dict(color=colors[i], width=line_width),
                    marker=dict(size=marker_size),
                    opacity=alpha
                ))
                
                if not data_length:  # Set data_length if not set yet
                    data_length = len(perplexities)
    else:
        # Single epoch visualization
        perplexities = pattern_data.get('perplexities', [])
        data_length = len(perplexities)
        x_positions = list(range(data_length))
        
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=perplexities,
            mode='lines+markers',
            name='Perplexity',
            line=dict(color=start_color, width=line_width),
            marker=dict(size=marker_size),
            opacity=alpha
        ))
    
    # Add invisible trace to force second x-axis to appear
    axis_x_positions = list(range(data_length)) if data_length > 0 else []
    if axis_x_positions:
        fig.add_trace(go.Scatter(
            x=axis_x_positions,
            y=[None] * len(axis_x_positions),  # Invisible data
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=0),  # Transparent markers
            showlegend=False,
            xaxis='x2',
            yaxis='y',
            hoverinfo='skip'
        ))
    
    # Get tokens for x-axis labels (but don't depend on them for plotting)
    tokens = pattern_data.get('tokens', [])
    
    # Configure axes using data_length
    x_range = [-1, data_length] if data_length > 0 else [-1, 101]
    
    xaxis_config = {
        "title": dict(text="Character Sequence", font=dict(size=font_size)),
        "type": "linear",
        "side": "bottom",
        "ticks": "outside",
        "ticklen": 5,
        "tickcolor": "black",
        "range": x_range,
        "tickfont": dict(size=font_size-2)
    }
    
    # Create x_positions for axis configuration
    axis_x_positions = list(range(data_length)) if data_length > 0 else []
    
    if tokens and len(tokens) == data_length:
        xaxis_config.update({
            "tickmode": "array",
            "tickvals": axis_x_positions,
            "ticktext": tokens,
            "tickangle": 0
        })
    
    # Secondary x-axis for token positions
    if data_length > 0:
        # Create labels: 1, 5, 10, 15, 20, 25...
        token_labels = [1] + [5 * i for i in range(1, (data_length // 5) + 2) if 5 * i <= data_length]
        # Convert to 0-based x-coordinates: 0, 4, 9, 14, 19, 24...
        token_tick_vals = [label - 1 for label in token_labels if label - 1 < data_length]
        token_tick_labels = [str(label) for label in token_labels if label - 1 < data_length]
    else:
        token_tick_vals = [0, 4, 9, 14, 19]
        token_tick_labels = ["1", "5", "10", "15", "20"]
    
    xaxis2_config = {
        "title": dict(text="Token Position", font=dict(size=font_size)),
        "type": "linear", 
        "side": "top",
        "overlaying": "x",
        "tickmode": "array",
        "tickvals": token_tick_vals,
        "ticktext": token_tick_labels,
        "showgrid": False,
        "showticklabels": True,
        "visible": True,
        "range": x_range,
        "ticks": "outside",
        "ticklen": 5,
        "tickcolor": "black",
        "tickfont": dict(size=font_size-2)
    }
    
    # Y-axis configuration
    all_perps = []
    if pattern_data.get('type') == 'multi_epoch':
        for epoch_perps in pattern_data['epoch_perplexities'].values():
            all_perps.extend(epoch_perps)
    else:
        all_perps.extend(pattern_data.get('perplexities', []))
    
    if use_log_scale and all_perps:
        min_perp = min(all_perps)
        max_perp = max(all_perps)
        
        min_power = int(np.floor(np.log10(min_perp))) if min_perp > 0 else -1
        max_power = int(np.ceil(np.log10(max_perp))) if max_perp > 0 else 4
        
        tick_values = [10**i for i in range(min_power, max_power + 1)]
        tick_labels = [f"10<sup>{i}</sup>" for i in range(min_power, max_power + 1)]
        
        if min_perp > 0 and max_perp > 0:
            y_min = np.log10(min_perp)
            y_max = np.log10(max_perp) + 0.3
        else:
            y_min = 0
            y_max = 4
            
        yaxis_config = {
            "title": dict(text="Perplexity (Log Scale)", font=dict(size=font_size)),
            "type": "log",
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": tick_labels,
            "showgrid": show_grid,
            "gridcolor": f"rgba(128, 128, 128, {grid_alpha})",
            "gridwidth": 1,
            "ticks": "outside",
            "ticklen": 5,
            "tickcolor": "black",
            "range": [y_min, y_max],
            "tickfont": dict(size=font_size-2)
        }
    else:
        if all_perps and min(all_perps) >= 0 and max(all_perps) > 0:
            y_min = min(all_perps)
            y_max = max(all_perps) * 1.2
        else:
            y_min = 0
            y_max = 100
            
        yaxis_config = {
            "title": dict(text="Perplexity", font=dict(size=font_size)),
            "type": "linear",
            "showgrid": show_grid,
            "gridcolor": f"rgba(128, 128, 128, {grid_alpha})",
            "gridwidth": 1,
            "ticks": "outside",
            "ticklen": 5,
            "tickcolor": "black",
            "range": [y_min, y_max],
            "tickfont": dict(size=font_size-2)
        }
    
    # Prepare shapes for custom grid lines
    shapes = []
    
    # Add thin black border around the plot
    shapes.append(dict(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1),
        fillcolor="rgba(0,0,0,0)"
    ))
    
    # Add vertical grid lines for each token (if grid is enabled)
    if show_grid and data_length > 0:
        axis_x_positions = list(range(data_length))
        for x in axis_x_positions:
            shapes.append(dict(
                type="line",
                xref="x", yref="paper",
                x0=x, y0=0, x1=x, y1=1,
                line=dict(color=f"rgba(128, 128, 128, {grid_alpha*0.5})", width=0.5)
            ))
    
    # Add logarithmic minor grid lines if using log scale and grid is enabled
    if use_log_scale and all_perps and show_grid:
        # Use the same y-axis range as the main axis
        if min(all_perps) > 0 and max(all_perps) > 0:
            y_min = min(all_perps)  # No bottom padding
            y_max = 10**(np.log10(max(all_perps)) + 0.08)  # 20% top padding
        else:
            y_min = 1.0  # Start at 10^0
            y_max = 10000
        
        # Add minor grid lines for each decade
        min_power = int(np.floor(np.log10(y_min))) if y_min > 0 else -1
        max_power = int(np.ceil(np.log10(y_max))) if y_max > 0 else 4
        
        for power in range(min_power, max_power + 1):
            decade_start = 10**power
            decade_end = 10**(power + 1)
            
            # Add minor lines at 2, 3, 4, 5, 6, 7, 8, 9 within each decade
            for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
                y_val = decade_start * multiplier
                if y_min <= y_val <= y_max:
                    shapes.append(dict(
                        type="line",
                        xref="paper", yref="y",
                        x0=0, y0=y_val, x1=1, y1=y_val,
                        line=dict(color=f"rgba(128, 128, 128, {grid_alpha*0.1})", width=0.5)
                    ))
    
    # Legend configuration
    legend_positions = {
        "Upper Right (Inside)": {"x": 0.99, "y": 0.97, "xanchor": "right", "yanchor": "top"},
        "Upper Left (Inside)": {"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
        "Lower Right (Inside)": {"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
        "Lower Left (Inside)": {"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
        "Center": {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle"},
        "Right (Outside)": {"x": 1.02, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
        "Left (Outside)": {"x": -0.02, "y": 0.5, "xanchor": "right", "yanchor": "middle"}
    }
    
    legend_config = legend_positions.get(legend_position, legend_positions["Upper Right (Inside)"])
    legend_config.update({
        "bgcolor": "rgba(255, 255, 255, 0.7)",
        "bordercolor": "black",
        "borderwidth": 1,
        "font": {"size": font_size}
    })
    
    # Adjust margins for outside legend positions
    margin_config = dict(t=80, b=60, l=40, r=20)
    if legend_position == "Right (Outside)":
        margin_config["r"] = 120
    elif legend_position == "Left (Outside)":
        margin_config["l"] = 120
    
    # Set background color
    plot_bg = "white"
    paper_bg = "white"
    if bg_color == "Light Gray":
        plot_bg = "#f8f9fa"
        paper_bg = "#f8f9fa"
    elif bg_color == "Transparent":
        plot_bg = "rgba(0,0,0,0)"
        paper_bg = "rgba(0,0,0,0)"
    
    fig.update_layout(
        title=dict(text="Perplexity Distribution", font=dict(size=font_size+2)),
        xaxis=xaxis_config,
        xaxis2=xaxis2_config,
        yaxis=yaxis_config,
        legend=legend_config,
        hovermode='x unified',
        width=width,
        height=height,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        shapes=shapes,
        margin=margin_config
    )
    
    return fig

def generate_python_code(pattern_index: int, analysis_data: Dict, viz_start: int, viz_end: int, 
                        use_log_scale: bool = True, alpha: float = 0.7, start_color: str = "#00FFBD", 
                        end_color: str = "#002BFF", width_inches: float = 14.0, height_inches: float = 4.0, 
                        marker_size: int = 4, dpi: int = 300, legend_position: str = "upper right", 
                        line_width: int = 2, show_grid: bool = True, grid_alpha: float = 0.3, 
                        font_size: int = 12, bg_color: str = "White") -> str:
    
    # Convert hex colors to RGB tuples for the code
    def hex_to_rgb_str(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"({r/255:.3f}, {g/255:.3f}, {b/255:.3f})"
    
    start_rgb = hex_to_rgb_str(start_color)
    end_rgb = hex_to_rgb_str(end_color)
    
    # Generate the code
    code = f'''import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_pattern_perplexity(pattern, viz_start, viz_end, start_rgb, end_rgb, width_inches):
    """
    Visualize perplexity data for a pattern across multiple epochs.
    
    Args:
        pattern: Pattern dictionary from the analysis data
        viz_start: Start index for visualization range
        viz_end: End index for visualization range
        start_rgb: RGB tuple for start color (r, g, b)
        end_rgb: RGB tuple for end color (r, g, b)
        width_inches: Width of the figure in inches
    """
    
    # Extract perplexity data for each epoch
    epoch_perplexities = {{}}
    epoch_data = pattern['full_document_epoch_data']
    available_epochs = sorted(epoch_data.keys(), key=lambda x: int(x))

    for epoch in available_epochs:
        epoch_tokens_data = epoch_data[epoch]
        # Extract perplexities from token data tuples (token, token_id, perplexity)
        perplexities = []
        for i in range(viz_start, viz_end):
            if i < len(epoch_tokens_data):
                _, _, perplexity = epoch_tokens_data[i]
                perplexities.append(perplexity)
        epoch_perplexities[int(epoch)] = perplexities

    # Extract tokens for the visualization range
    tokens = pattern['full_document_tokens'][viz_start:viz_end]
    epoch_numbers = [int(e) for e in available_epochs]

    # Color gradient function
    def generate_color_gradient(n_colors, start_rgb, end_rgb):
        if n_colors <= 1:
            return [start_rgb]
        colors = []
        for i in range(n_colors):
            ratio = i / (n_colors - 1)
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            colors.append((r, g, b))
        return colors

    # Format epoch names
    def format_epoch_name(epoch_num):
        if epoch_num == -1:
            return "Base"
        else:
            return f"Epoch {{epoch_num + 1}}"

    # Create the plot
    fig, ax = plt.subplots(figsize=(width_inches, {height_inches * 1.2:.1f}), dpi={dpi})

    # Generate colors
    colors = generate_color_gradient(len(epoch_numbers), start_rgb, end_rgb)

    # Plot each epoch
    for i, epoch_num in enumerate(epoch_numbers):
        if epoch_num in epoch_perplexities:
            perplexities = epoch_perplexities[epoch_num]
            x_positions = list(range(len(perplexities)))
            ax.plot(x_positions, perplexities, 
                   color=colors[i], linewidth={line_width}, marker='o', markersize={marker_size * 0.75:.1f},
                   alpha={alpha * 0.7:.2f}, label=format_epoch_name(epoch_num))

    # Set up axes
    data_length = len(tokens)
    x_range = [-1, data_length]
    ax.set_xlim(x_range)
    ax.set_xlabel("Character Sequence", fontsize={font_size + 2}, fontweight='bold')

    # Set x-axis ticks to show tokens
    if tokens:
        ax.set_xticks(list(range(data_length)))
        ax.set_xticklabels(tokens, rotation=0, fontsize={font_size})

    # Secondary x-axis for token positions
    ax2 = ax.twiny()
    if data_length > 0:
        token_labels = [1] + [5 * i for i in range(1, (data_length // 5) + 2) if 5 * i <= data_length]
        token_tick_vals = [label - 1 for label in token_labels if label - 1 < data_length]
        token_tick_labels = [str(label) for label in token_labels if label - 1 < data_length]
    else:
        token_tick_vals = [0, 4, 9, 14, 19]
        token_tick_labels = ["1", "5", "10", "15", "20"]

    ax2.set_xlim(x_range)
    ax2.set_xticks(token_tick_vals)
    ax2.set_xticklabels(token_tick_labels, fontsize={font_size})
    ax2.set_xlabel("Token Position", fontsize={font_size + 2}, fontweight='bold')

    # Y-axis setup
    all_perps = []
    for perps in epoch_perplexities.values():
        all_perps.extend(perps)

    if {str(use_log_scale).lower()} and all_perps:
        ax.set_yscale('log')
        ax.set_ylabel("Perplexity (Log Scale)", fontsize={font_size + 2}, fontweight='bold')
        if min(all_perps) > 0 and max(all_perps) > 0:
            y_min = min(all_perps)
            y_max = max(all_perps) * 3
            ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylabel("Perplexity", fontsize={font_size + 2}, fontweight='bold')
        if all_perps and min(all_perps) >= 0:
            y_min = min(all_perps)
            y_max = max(all_perps) * 1.2
            ax.set_ylim([y_min, y_max])

    # Grid and legend
    ax.grid({str(show_grid).lower()}, alpha={grid_alpha:.2f}, linewidth=1)
    legend = ax.legend(loc='{legend_position.lower().replace(" (inside)", "").replace(" (outside)", "")}', 
                      fontsize={font_size}, frameon=True, facecolor="white", 
                      edgecolor="black", framealpha=0.7)

    if legend.get_frame():
        legend.get_frame().set_linewidth(1)

    plt.tight_layout(pad=2.0)

    # Background color
    face_color = 'white'
    bg_color_setting = '{bg_color}'
    if bg_color_setting == "Light Gray":
        face_color = '#f8f9fa'
    elif bg_color_setting == "Transparent":
        face_color = 'none'

    # Save or show the plot
    plt.savefig('pattern_visualization.png', format='png', dpi={dpi}, 
                bbox_inches='tight', facecolor=face_color, edgecolor='none', pad_inches=0.3)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load the JSON file
    with open('your_pattern_analysis.json', 'r') as f:
        analysis_data = json.load(f)
    
    # Example: visualize pattern {pattern_index} with range {viz_start}:{viz_end}
    pattern_index = {pattern_index}
    pattern = analysis_data['patterns'][pattern_index]
    viz_start = {viz_start}
    viz_end = {viz_end}
    start_rgb = {start_rgb}
    end_rgb = {end_rgb}
    width_inches = {width_inches:.1f}
    
    visualize_pattern_perplexity(pattern, viz_start, viz_end, start_rgb, end_rgb, width_inches)'''

    return code

def create_matplotlib_png(pattern_data: Dict, use_log_scale: bool = True, 
                         alpha: float = 0.7, start_color: str = "#00FFBD", end_color: str = "#002BFF",
                         width_inches: float = 14.0, height_inches: float = 4.0, marker_size: int = 4, 
                         dpi: int = 300, legend_position: str = "upper right", line_width: int = 2,
                         show_grid: bool = True, grid_alpha: float = 0.3, font_size: int = 12,
                         bg_color: str = "White") -> BytesIO:
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return tuple(c / 255.0 for c in rgb)
    
    def generate_color_gradient(n_colors, start_hex, end_hex):
        if n_colors <= 1:
            return [start_hex]
        
        start_rgb = hex_to_rgb(start_hex)
        end_rgb = hex_to_rgb(end_hex)
        
        colors = []
        for i in range(n_colors):
            ratio = i / (n_colors - 1)
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            colors.append((r, g, b))
        return colors
    
    def format_epoch_name(epoch_num):
        if epoch_num == -1:
            return "Base"
        else:
            return f"Epoch {epoch_num + 1}"
    
    # Set up figure
    fig_width = width_inches
    fig_height = height_inches * 1.2
    marker_size = marker_size * 0.75
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Get data and determine length
    tokens = pattern_data.get('tokens', [])
    data_length = 0
    
    if pattern_data.get('type') == 'multi_epoch':
        # Multi-epoch visualization
        epoch_perps = pattern_data.get('epoch_perplexities', {})
        epoch_numbers = pattern_data.get('epoch_numbers', [])
        
        if epoch_perps and epoch_numbers:
            # Get data length from first available epoch
            first_epoch = epoch_numbers[0]
            if first_epoch in epoch_perps:
                data_length = len(epoch_perps[first_epoch])
        
        colors = generate_color_gradient(len(epoch_numbers), start_color, end_color)
        
        for i, epoch_num in enumerate(epoch_numbers):
            if epoch_num in epoch_perps:
                perplexities = epoch_perps[epoch_num]
                x_positions = list(range(len(perplexities)))
                ax.plot(x_positions, perplexities, 
                       color=colors[i], linewidth=line_width, marker='o', markersize=marker_size,
                       alpha=alpha*0.7, label=format_epoch_name(epoch_num))
    else:
        # Single epoch visualization
        perplexities = pattern_data.get('perplexities', [])
        data_length = len(perplexities)
        x_positions = list(range(data_length))
        start_rgb = hex_to_rgb(start_color)
        
        ax.plot(x_positions, perplexities, 
               color=start_rgb, linewidth=line_width, marker='o', markersize=marker_size,
               alpha=alpha, label='Perplexity')
    
    # Set up axes
    x_range = [-1, data_length] if data_length > 0 else [-1, 101]
    ax.set_xlim(x_range)
    ax.set_xlabel("Character Sequence", fontsize=font_size+2, fontweight='bold')
    
    # Create axis positions for configuration
    axis_x_positions = list(range(data_length)) if data_length > 0 else []
    
    if tokens and len(tokens) == data_length:
        ax.set_xticks(axis_x_positions)
        ax.set_xticklabels(tokens, rotation=0, fontsize=font_size)
    
    # Set up secondary x-axis
    ax2 = ax.twiny()
    if data_length > 0:
        # Create labels: 1, 5, 10, 15, 20, 25...
        token_labels = [1] + [5 * i for i in range(1, (data_length // 5) + 2) if 5 * i <= data_length]
        # Convert to 0-based x-coordinates: 0, 4, 9, 14, 19, 24...
        token_tick_vals = [label - 1 for label in token_labels if label - 1 < data_length]
        token_tick_labels = [str(label) for label in token_labels if label - 1 < data_length]
    else:
        token_tick_vals = [0, 4, 9, 14, 19]
        token_tick_labels = ["1", "5", "10", "15", "20"]
    
    ax2.set_xlim(x_range)
    ax2.set_xticks(token_tick_vals)
    ax2.set_xticklabels(token_tick_labels, fontsize=font_size)
    ax2.set_xlabel("Token Position", fontsize=font_size+2, fontweight='bold')
    
    # Set up y-axis
    all_perps = []
    if pattern_data.get('type') == 'multi_epoch':
        for epoch_perps in pattern_data['epoch_perplexities'].values():
            all_perps.extend(epoch_perps)
    else:
        all_perps.extend(pattern_data.get('perplexities', []))
    
    if use_log_scale and all_perps:
        ax.set_yscale('log')
        ax.set_ylabel("Perplexity (Log Scale)", fontsize=font_size+2, fontweight='bold')
        
        if min(all_perps) > 0 and max(all_perps) > 0:
            y_min = min(all_perps)
            y_max = max(all_perps) * 3
            ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylabel("Perplexity", fontsize=font_size+2, fontweight='bold')
        if all_perps and min(all_perps) >= 0:
            y_min = min(all_perps)
            y_max = max(all_perps) * 1.2
            ax.set_ylim([y_min, y_max])
    
    # Add grid and legend
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=1)
    else:
        ax.grid(False)
    
    legend = ax.legend(loc=legend_position, fontsize=font_size, frameon=True, 
                      facecolor="white", edgecolor="black", framealpha=0.7)
    
    if legend.get_frame():
        legend.get_frame().set_linewidth(1)
    
    plt.tight_layout(pad=2.0)
    
    # Set background color
    face_color = 'white'
    if bg_color == "Light Gray":
        face_color = '#f8f9fa'
    elif bg_color == "Transparent":
        face_color = 'none'
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor=face_color, edgecolor='none', pad_inches=0.3)
    buffer.seek(0)
    plt.close(fig)
    
    return buffer

def main():
    st.title("🔍 Pattern Viewer")
    st.markdown("Visualize learning patterns from `find_significant_patterns.py` output")
    
    # Sidebar for global controls
    with st.sidebar:
        st.header("🎛️ Global Settings")
        
        # Context settings
        st.subheader("Context Settings")
        left_context = st.number_input("Left Context (characters)", min_value=0, max_value=200, value=30, step=5,
                                      help="Number of characters to show before each pattern")
        right_context = st.number_input("Right Context (characters)", min_value=0, max_value=200, value=30, step=5,
                                       help="Number of characters to show after each pattern")
        
        # Visualization defaults
        st.subheader("Visualization Defaults")
        default_viz_left = st.number_input("Default Viz Left Extend", min_value=0, max_value=100, value=30, step=5,
                                          help="Default characters to extend visualization to the left of pattern")
        default_viz_right = st.number_input("Default Viz Right Extend", min_value=0, max_value=100, value=30, step=5,
                                           help="Default characters to extend visualization to the right of pattern")
        
        # Pattern highlighting
        st.subheader("Pattern Highlighting")
        add_asterisks = st.checkbox("Add ** around pattern", value=False,
                                   help="Add asterisks (**pattern**) around the highlighted pattern text")
        
        # Pattern filtering
        st.subheader("Pattern Filtering")
        filter_chars = st.text_input("Hide patterns containing these characters:", value="",
                                    help="Enter characters (e.g., '123abc'). Patterns containing any of these characters will be hidden.")
        filter_chars = filter_chars.strip()
    
    # File upload
    st.header("📁 Upload Pattern Analysis")
    uploaded_file = st.file_uploader(
        "Choose a JSON file from find_significant_patterns.py",
        type=['json'],
        help="Upload the JSON output from find_significant_patterns.py script"
    )
    
    if not uploaded_file:
        st.info("👆 Please upload a JSON file to begin")
        st.markdown("""
        ### How to generate the input file:
        ```bash
        python experiments/find_significant_patterns.py \\
            --input-directory app/training_output_2e-5/ \\
            --output results/learning_patterns.json \\
            --mode top-bottom \\
            --top-n 100
        ```
        """)
        return
    
    # Load and validate JSON
    analysis_data, error = load_pattern_analysis(uploaded_file)
    if error:
        st.error(f"❌ {error}")
        return
    
    st.success("✅ Pattern analysis loaded successfully!")
    
    # Display analysis summary
    st.header("📊 Analysis Summary")
    
    statistics = analysis_data["statistics"]
    patterns = analysis_data["patterns"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patterns", f"{statistics['total_patterns']:,}")
    with col2:
        st.metric("Patterns in Output", f"{len(patterns):,}")
    with col3:
        st.metric("Improvements", f"{statistics.get('total_improvements', 0):,}")
    with col4:
        st.metric("Deteriorations", f"{statistics.get('total_deteriorations', 0):,}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pattern Length", statistics['pattern_length'])
    with col2:
        st.metric("Mean Improvement", f"{statistics['mean_log_improvement']:.4f}")
    with col3:
        st.metric("Analysis Method", statistics.get('method', 'simple_ranking'))
    
    # Pattern list
    st.header("📋 Learning Patterns")
    
    if not patterns:
        st.warning("No patterns found in this analysis.")
        return
    
    # Filter patterns based on filter characters
    visible_patterns = []
    for i, pattern in enumerate(patterns):
        if is_pattern_visible(pattern, filter_chars):
            # Add the original index and visible flag to the pattern
            pattern_with_index = pattern.copy()
            pattern_with_index['original_index'] = i
            pattern_with_index['visible'] = True
            visible_patterns.append(pattern_with_index)
    
    # Show filtering info
    if filter_chars:
        total_patterns = len(patterns)
        visible_count = len(visible_patterns)
        hidden_count = total_patterns - visible_count
        st.info(f"🔍 Filter: '{filter_chars}' | Showing {visible_count}/{total_patterns} patterns ({hidden_count} hidden)")
    
    if not visible_patterns:
        st.warning("No patterns match the current filter. Try removing some filter characters.")
        return
    
    # Initialize session state for selected pattern
    if 'selected_pattern_idx' not in st.session_state:
        st.session_state.selected_pattern_idx = None
    
    # Display visible patterns as a list with visualize buttons
    for i, pattern in enumerate(visible_patterns):
        with st.container():
            # Create columns for pattern info and button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Pattern header
                direction_icon = "📉" if pattern['improvement'] > 0 else "📈"
                direction_desc = "Improvement" if pattern['improvement'] > 0 else "Deterioration"
                original_idx = pattern.get('original_index', i)
                
                st.markdown(f"""
                **{direction_icon} Pattern {original_idx+1}** - {direction_desc}  
                📍 **{pattern['text_id']}** [{pattern['start_idx']}:{pattern['end_idx']}]  
                📊 **Improvement:** {pattern['improvement']:.4f} | **Relative:** {pattern['relative_improvement']*100:.1f}%
                """)
                
                # Show pattern with context
                before_pattern, pattern_text, after_pattern = get_pattern_with_context(pattern, left_context, right_context)
                
                # Add asterisks if enabled
                display_pattern = f"**{pattern_text}**" if add_asterisks else pattern_text
                
                # Escape HTML to prevent formatting issues
                before_escaped = html.escape(before_pattern)
                pattern_escaped = html.escape(display_pattern)
                after_escaped = html.escape(after_pattern)
                
                # Format with highlighting
                context_html = f"""
                <div style="font-family: monospace; padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin: 5px 0;">
                    <span style="color: #666;">{before_escaped}</span><span style="background-color: #ffeb3b; font-weight: bold;">{pattern_escaped}</span><span style="color: #666;">{after_escaped}</span>
                </div>
                """
                st.markdown(context_html, unsafe_allow_html=True)
            
            with col2:
                # Visualize button
                original_idx = pattern.get('original_index', i)
                if st.button(f"🎯 Visualize", key=f"viz_btn_{original_idx}", type="primary"):
                    st.session_state.selected_pattern_idx = original_idx
                    st.rerun()
            
            st.divider()
    
    # Show visualization section if a pattern is selected
    if st.session_state.selected_pattern_idx is not None:
        selected_pattern = patterns[st.session_state.selected_pattern_idx]
        
        st.header(f"🎯 Pattern Visualization - Pattern {st.session_state.selected_pattern_idx + 1}")
        
        # Pattern details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text ID", selected_pattern['text_id'])
        with col2:
            direction_desc = "Improvement" if selected_pattern['improvement'] > 0 else "Deterioration"
            st.metric("Direction", direction_desc)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Early Perplexity", f"{selected_pattern['early_perplexity']:.2f}")
        with col2:
            st.metric("Late Perplexity", f"{selected_pattern['late_perplexity']:.2f}")
        with col3:
            st.metric("Log Improvement", f"{selected_pattern['improvement']:.4f}")
        with col4:
            st.metric("Relative Improvement", f"{selected_pattern['relative_improvement']*100:.1f}%")
        
        # Visualization controls
        st.subheader("🎨 Visualization Settings")
        
        # Get pattern boundaries
        pattern_start = selected_pattern['start_idx']
        pattern_end = selected_pattern['end_idx']
        text_length = len(selected_pattern['full_document_text'])
        
        # Calculate default visualization range
        viz_start = max(0, pattern_start - default_viz_left)
        viz_end = min(text_length, pattern_end + default_viz_right)
        
        # Initialize visualization state in session state
        if 'visualization_generated' not in st.session_state:
            st.session_state.visualization_generated = False
        if 'viz_data' not in st.session_state:
            st.session_state.viz_data = None
        if 'viz_settings' not in st.session_state:
            st.session_state.viz_settings = {}
        
        # Visualization form
        with st.form("visualization_form", clear_on_submit=False):
            st.subheader("📏 Visualization Range")
            
            col1, col2 = st.columns(2)
            with col1:
                viz_start_input = st.number_input(
                    "Visualization Start", 
                    min_value=0, 
                    max_value=text_length-1, 
                    value=viz_start,
                    help=f"Pattern starts at index {pattern_start}"
                )
            with col2:
                viz_end_input = st.number_input(
                    "Visualization End", 
                    min_value=viz_start_input + 1,
                    max_value=text_length, 
                    value=viz_end,
                    help=f"Pattern ends at index {pattern_end}"
                )
            
            st.subheader("🎨 Plot Styling")
            
            col1, col2 = st.columns(2)
            with col1:
                use_log_scale = st.checkbox("Use Log Scale", value=True)
                alpha_value = st.slider("Line Transparency", 0.1, 1.0, 0.7, step=0.1)
                line_width = st.slider("Line Width", 1, 5, 2, step=1)
            with col2:
                start_color = st.color_picker("Start Color", "#00FFBD", key="start_color_picker")
                end_color = st.color_picker("End Color", "#002BFF", key="end_color_picker")
                show_grid = st.checkbox("Show Grid Lines", value=True)
            
            col1, col2 = st.columns(2)
            with col1:
                grid_alpha = st.slider("Grid Transparency", 0.1, 1.0, 0.3, step=0.1)
                font_size = st.slider("Font Size", 8, 20, 12, step=1)
            with col2:
                bg_color = st.selectbox("Background", ["White", "Light Gray", "Transparent"])
                export_dpi = st.selectbox("Export DPI", [150, 300, 600, 1200], index=1)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                legend_options = [
                    "Upper Right (Inside)", "Upper Left (Inside)", "Lower Right (Inside)", 
                    "Lower Left (Inside)", "Center", "Right (Outside)", "Left (Outside)"
                ]
                legend_position = st.selectbox("Legend Position", legend_options)
            with col2:
                graph_width_inches = st.slider("Width (inches)", 4.0, 20.0, 14.0, step=0.5)
                graph_width = int(graph_width_inches * 100)
            with col3:
                graph_height_inches = st.slider("Height (inches)", 3.0, 15.0, 4.0, step=0.5)
                graph_height = int(graph_height_inches * 100)
            
            marker_size = st.slider("Marker Size", 1, 15, 4, step=1)
            
            submitted = st.form_submit_button("🔄 Generate Visualization", type="primary")
        
        # Generate visualization
        if submitted:
            if viz_start_input >= viz_end_input:
                st.error("Start index must be less than end index!")
            elif viz_end_input > text_length:
                st.error(f"End index cannot exceed text length ({text_length})!")
            else:
                # Extract visualization data
                viz_data, viz_error = extract_visualization_data(selected_pattern, viz_start_input, viz_end_input)
                
                if viz_error:
                    st.error(f"❌ {viz_error}")
                    st.session_state.visualization_generated = False
                else:
                    # Store visualization data and settings in session state
                    st.session_state.viz_data = viz_data
                    st.session_state.viz_start = viz_start_input
                    st.session_state.viz_end = viz_end_input
                    st.session_state.viz_settings = {
                        'use_log_scale': use_log_scale,
                        'alpha_value': alpha_value,
                        'line_width': line_width,
                        'start_color': start_color,
                        'end_color': end_color,
                        'show_grid': show_grid,
                        'grid_alpha': grid_alpha,
                        'font_size': font_size,
                        'bg_color': bg_color,
                        'export_dpi': export_dpi,
                        'legend_position': legend_position,
                        'graph_width': graph_width,
                        'graph_height': graph_height,
                        'graph_width_inches': graph_width_inches,
                        'graph_height_inches': graph_height_inches,
                        'marker_size': marker_size
                    }
                    st.session_state.visualization_generated = True
        
        # Display visualization if it has been generated
        if st.session_state.visualization_generated and st.session_state.viz_data:
            viz_data = st.session_state.viz_data
            settings = st.session_state.viz_settings
            
            # Show epoch information
            epochs_available = viz_data['epoch_numbers']
            st.info(f"📅 Available epochs: {epochs_available}")
            
            # Get current color values for the interactive plot
            current_start_color = st.session_state.get('start_color_picker', settings['start_color'])
            current_end_color = st.session_state.get('end_color_picker', settings['end_color'])
            
            # Create and display plot
            fig = plot_perplexity_distribution(
                viz_data, 
                settings['use_log_scale'], 
                settings['alpha_value'], 
                current_start_color, 
                current_end_color,
                settings['graph_width'], 
                settings['graph_height'], 
                settings['marker_size'], 
                settings['legend_position'],
                settings['line_width'],
                settings['show_grid'],
                settings['grid_alpha'],
                settings['font_size'],
                settings['bg_color']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.subheader("📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Generate High-Res PNG", type="primary", key="png_btn"):
                    try:
                        with st.spinner("Generating PNG and Python code..."):
                            # Get current color values from session state (in case user changed them without resubmitting form)
                            current_start_color = st.session_state.get('start_color_picker', settings['start_color'])
                            current_end_color = st.session_state.get('end_color_picker', settings['end_color'])
                            
                            png_buffer = create_matplotlib_png(
                                viz_data, 
                                settings['use_log_scale'], 
                                settings['alpha_value'], 
                                current_start_color, 
                                current_end_color,
                                settings['graph_width_inches'], 
                                settings['graph_height_inches'], 
                                settings['marker_size'], 
                                dpi=settings['export_dpi'],
                                legend_position=settings['legend_position'].lower().replace(" (inside)", "").replace(" (outside)", ""),
                                line_width=settings['line_width'],
                                show_grid=settings['show_grid'],
                                grid_alpha=settings['grid_alpha'],
                                font_size=settings['font_size'],
                                bg_color=settings['bg_color']
                            )
                            
                            # Generate Python code
                            python_code = generate_python_code(
                                pattern_index=st.session_state.selected_pattern_idx,
                                analysis_data=analysis_data,
                                viz_start=st.session_state.viz_start,
                                viz_end=st.session_state.viz_end,
                                use_log_scale=settings['use_log_scale'], 
                                alpha=settings['alpha_value'], 
                                start_color=current_start_color, 
                                end_color=current_end_color,
                                width_inches=settings['graph_width_inches'], 
                                height_inches=settings['graph_height_inches'], 
                                marker_size=settings['marker_size'], 
                                dpi=settings['export_dpi'],
                                legend_position=settings['legend_position'],
                                line_width=settings['line_width'],
                                show_grid=settings['show_grid'],
                                grid_alpha=settings['grid_alpha'],
                                font_size=settings['font_size'],
                                bg_color=settings['bg_color']
                            )
                            
                            filename = f"pattern_{selected_pattern['text_id']}_{selected_pattern['start_idx']}_{selected_pattern['end_idx']}.png"
                            
                            st.session_state['png_data'] = png_buffer.getvalue()
                            st.session_state['png_filename'] = filename
                            st.session_state['python_code'] = python_code
                            
                            st.success("✅ PNG and Python code generated!")
                    except Exception as e:
                        st.error(f"Error creating PNG: {e}")
            
            with col2:
                if 'png_data' in st.session_state:
                    st.download_button(
                        label="📁 Download PNG File",
                        data=st.session_state['png_data'],
                        file_name=st.session_state['png_filename'],
                        mime="image/png",
                        key="download_btn"
                    )
                else:
                    st.info("Generate PNG first to enable download")
            
            # Download Python code
            if 'python_code' in st.session_state:
                st.subheader("🐍 Python Code Download")
                col1, col2 = st.columns(2)
                with col1:
                    python_filename = f"pattern_{selected_pattern['text_id']}_{selected_pattern['start_idx']}_{selected_pattern['end_idx']}_visualization.py"
                    st.download_button(
                        label="📄 Download Python Code",
                        data=st.session_state['python_code'],
                        file_name=python_filename,
                        mime="text/plain",
                        key="download_python_btn",
                        type="secondary"
                    )
                with col2:
                    st.info("💡 Update the JSON file path in the downloaded code and run it to recreate the visualization.")
            
            # Pattern statistics
            st.subheader("📊 Visualization Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                all_perps = []
                for perps in viz_data['epoch_perplexities'].values():
                    all_perps.extend(perps)
                viz_length = len(viz_data['tokens'])
                st.metric("Visualization Length", f"{viz_length} characters")
            with col2:
                st.metric("Available Epochs", len(epochs_available))
            with col3:
                if all_perps:
                    st.metric("Perplexity Range", f"{min(all_perps):.1f} - {max(all_perps):.1f}")
            with col4:
                if len(epochs_available) >= 2:
                    first_epoch = min(epochs_available)
                    last_epoch = max(epochs_available)
                    st.metric("Epoch Range", f"{first_epoch} to {last_epoch}")
        
        # Add a button to clear selection
        if st.button("🔄 Clear Selection", type="secondary"):
            st.session_state.selected_pattern_idx = None
            st.session_state.visualization_generated = False
            st.session_state.viz_data = None
            st.session_state.viz_settings = {}
            # Clear PNG data and Python code when clearing selection
            if 'png_data' in st.session_state:
                del st.session_state['png_data']
            if 'png_filename' in st.session_state:
                del st.session_state['png_filename']
            if 'python_code' in st.session_state:
                del st.session_state['python_code']
            st.rerun()

if __name__ == "__main__":
    main() 