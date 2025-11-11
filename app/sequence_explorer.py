#!/usr/bin/env python3
"""Streamlit app for exploring text sequences from perplexity data."""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import re
from pathlib import Path
from qhchina.helpers import load_fonts
load_fonts()

# Configure page
st.set_page_config(
    page_title="Sequence Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_epoch_files(uploaded_files):
    epoch_data = {}
    errors = []
    
    for uploaded_file in uploaded_files:
        try:
            content = json.loads(uploaded_file.getvalue().decode("utf-8"))
            filename = uploaded_file.name
            
            # Extract epoch number from filename
            match = re.search(r'epoch_(-?\d+)_perplexities', filename)
            if match:
                epoch_num = int(match.group(1))
            else:
                errors.append(f"Could not extract epoch number from {filename}")
                continue
            
            # Validate content structure
            if not isinstance(content, list):
                errors.append(f"Invalid format in {filename}: expected list of documents")
                continue
            
            # Convert to easier lookup format
            epoch_dict = {}
            for doc in content:
                if 'text_id' not in doc or 'token_data' not in doc:
                    continue
                
                text_id = doc['text_id']
                
                # Extract data from token_data tuples (token, token_id, perplexity)
                tokens = []
                perplexities = []
                for token_data in doc['token_data']:
                    if len(token_data) >= 3:
                        token, token_id, perplexity = token_data[:3]
                        tokens.append(token)
                        perplexities.append(perplexity)
                
                epoch_dict[text_id] = {
                    'text': doc.get('text', ''),
                    'tokens': tokens,
                    'perplexities': perplexities,
                    'token_data': doc['token_data'],
                    'metadata': doc.get('metadata', {})
                }
            
            epoch_data[epoch_num] = epoch_dict
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error in {uploaded_file.name}: {e}")
        except Exception as e:
            errors.append(f"Error loading {uploaded_file.name}: {e}")
    
    return epoch_data, errors

def extract_sequence_data(epoch_data: Dict, text_id: str, start_idx: int, end_idx: int) -> Tuple[Dict, Optional[str]]:
    try:
        # Get available epochs
        available_epochs = list(epoch_data.keys())
        available_epochs.sort()
        
        # Check if text_id exists in all epochs
        for epoch in available_epochs:
            if text_id not in epoch_data[epoch]:
                return None, f"Text ID '{text_id}' not found in epoch {epoch}"
        
        # Extract sequence data for each epoch
        epoch_perplexities = {}
        sequence_tokens = None
        
        for epoch in available_epochs:
            text_data = epoch_data[epoch][text_id]
            
            # Validate range (end_idx is inclusive)
            if end_idx >= len(text_data['tokens']):
                return None, f"End index {end_idx} exceeds document length {len(text_data['tokens'])-1} in epoch {epoch}"
            
            if start_idx > end_idx:
                return None, f"Start index {start_idx} must be less than or equal to end index {end_idx}"
            
            # Extract sequence perplexities (end_idx is inclusive)
            sequence_perps = text_data['perplexities'][start_idx:end_idx+1]
            epoch_perplexities[epoch] = sequence_perps
            
            # Get tokens (should be same across epochs, end_idx is inclusive)
            if sequence_tokens is None:
                sequence_tokens = text_data['tokens'][start_idx:end_idx+1]
        
        return {
            'type': 'multi_epoch',
            'epoch_perplexities': epoch_perplexities,
            'epoch_numbers': available_epochs,
            'tokens': sequence_tokens,
            'perplexity_positions': list(range(start_idx, end_idx+1))
        }, None
        
    except Exception as e:
        return None, f"Error extracting sequence data: {str(e)}"

def get_sequence_with_context(epoch_data: Dict, text_id: str, start_idx: int, end_idx: int, 
                             left_context: int, right_context: int) -> Tuple[str, str, str]:
    # Use first epoch for text (should be same across epochs)
    first_epoch = min(epoch_data.keys())
    full_text = epoch_data[first_epoch][text_id]['text']
    
    # Calculate context boundaries (end_idx is inclusive)
    context_start = max(0, start_idx - left_context)
    context_end = min(len(full_text), end_idx + 1 + right_context)
    
    # Extract text with context (end_idx is inclusive)
    before_sequence = full_text[context_start:start_idx]
    sequence_text = full_text[start_idx:end_idx+1]
    after_sequence = full_text[end_idx+1:context_end]
    
    return before_sequence, sequence_text, after_sequence

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
        title=dict(text="Sequence Perplexity Distribution", font=dict(size=font_size+2)),
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
    
    # Set matplotlib backend to Agg to avoid display issues
    import matplotlib
    matplotlib.use('Agg')
    
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

def generate_python_code(text_id: str, start_idx: int, end_idx: int, start_color: str, end_color: str,
                        use_log_scale: bool = True, alpha: float = 0.7, width_inches: float = 14.0, 
                        height_inches: float = 4.0, marker_size: int = 4, dpi: int = 300, 
                        legend_position: str = "upper right", line_width: int = 2, 
                        show_grid: bool = True, grid_alpha: float = 0.3, font_size: int = 12, 
                        bg_color: str = "White") -> str:
    
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
import re
from pathlib import Path

def visualize_sequence_perplexity(text_id, start_idx, end_idx, start_rgb, end_rgb, width_inches):
    """
    Visualize perplexity data for a sequence across multiple epochs.
    
    Args:
        text_id: Text document ID to analyze
        start_idx: Start index for sequence (inclusive)
        end_idx: End index for sequence (inclusive)
        start_rgb: RGB tuple for start color (r, g, b)
        end_rgb: RGB tuple for end color (r, g, b)
        width_inches: Width of the figure in inches
    """
    
    # Load epoch JSON files
    epoch_data = {{}}
    
    # You need to update this path to point to your epoch JSON files directory
    epoch_files_directory = "path/to/your/epoch/files"
    
    # Look for epoch files in the directory
    epoch_files = list(Path(epoch_files_directory).glob("epoch_*_perplexities.json"))
    
    for epoch_file in epoch_files:
        # Extract epoch number from filename
        match = re.search(r'epoch_(-?\\d+)_perplexities', epoch_file.name)
        if match:
            epoch_num = int(match.group(1))
            
            # Load the JSON file
            with open(epoch_file, 'r') as f:
                content = json.load(f)
            
            # Convert to easier lookup format
            epoch_dict = {{}}
            for doc in content:
                if 'text_id' not in doc or 'token_data' not in doc:
                    continue
                
                doc_text_id = doc['text_id']
                
                # Extract data from token_data tuples (token, token_id, perplexity)
                tokens = []
                perplexities = []
                for token_data in doc['token_data']:
                    if len(token_data) >= 3:
                        token, token_id, perplexity = token_data[:3]
                        tokens.append(token)
                        perplexities.append(perplexity)
                
                epoch_dict[doc_text_id] = {{
                    'text': doc.get('text', ''),
                    'tokens': tokens,
                    'perplexities': perplexities
                }}
            
            epoch_data[epoch_num] = epoch_dict
    
    # Extract sequence data for each epoch
    available_epochs = sorted(epoch_data.keys())
    epoch_perplexities = {{}}
    sequence_tokens = None
    
    for epoch in available_epochs:
        if text_id not in epoch_data[epoch]:
            print(f"Warning: Text ID '{{text_id}}' not found in epoch {{epoch}}")
            continue
        
        text_data = epoch_data[epoch][text_id]
        
        # Validate range (end_idx is inclusive)
        if end_idx >= len(text_data['tokens']):
            print(f"Warning: End index {{end_idx}} exceeds document length {{len(text_data['tokens'])-1}} in epoch {{epoch}}")
            continue
        
        # Extract sequence perplexities (end_idx is inclusive)
        sequence_perps = text_data['perplexities'][start_idx:end_idx+1]
        epoch_perplexities[epoch] = sequence_perps
        
        # Get tokens (should be same across epochs, end_idx is inclusive)
        if sequence_tokens is None:
            sequence_tokens = text_data['tokens'][start_idx:end_idx+1]
    
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
    colors = generate_color_gradient(len(available_epochs), start_rgb, end_rgb)

    # Plot each epoch
    for i, epoch_num in enumerate(available_epochs):
        if epoch_num in epoch_perplexities:
            perplexities = epoch_perplexities[epoch_num]
            x_positions = list(range(len(perplexities)))
            ax.plot(x_positions, perplexities, 
                   color=colors[i], linewidth={line_width}, marker='o', markersize={marker_size * 0.75:.1f},
                   alpha={alpha * 0.7:.2f}, label=format_epoch_name(epoch_num))

    # Set up axes
    data_length = len(sequence_tokens) if sequence_tokens else 0
    x_range = [-1, data_length]
    ax.set_xlim(x_range)
    ax.set_xlabel("Character Sequence", fontsize={font_size + 2}, fontweight='bold')

    # Set x-axis ticks to show tokens
    if sequence_tokens:
        ax.set_xticks(list(range(data_length)))
        ax.set_xticklabels(sequence_tokens, rotation=0, fontsize={font_size})

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
    plt.savefig('sequence_visualization.png', format='png', dpi={dpi}, 
                bbox_inches='tight', facecolor=face_color, edgecolor='none', pad_inches=0.3)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example: visualize text_id '{text_id}' sequence [{start_idx}:{end_idx}]
    text_id = '{text_id}'
    start_idx = {start_idx}
    end_idx = {end_idx}
    start_rgb = {start_rgb}
    end_rgb = {end_rgb}
    width_inches = {width_inches:.1f}
    
    visualize_sequence_perplexity(text_id, start_idx, end_idx, start_rgb, end_rgb, width_inches)'''

    return code

def main():
    st.title("🔬 Sequence Explorer")
    st.markdown("Freely explore any text sequence from raw epoch perplexity data")
    
    # File upload
    st.header("📁 Upload Epoch Files")
    uploaded_files = st.file_uploader(
        "Choose epoch JSON files from calculate_perplexity.py",
        type=['json'],
        accept_multiple_files=True,
        help="Upload multiple epoch files (e.g., epoch_-1_perplexities.json, epoch_0_perplexities.json, etc.)"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload epoch JSON files to begin exploration")
        st.markdown("""
        ### Expected Files:
        Upload multiple JSON files with names like:
        - `epoch_-1_perplexities.json`
        - `epoch_0_perplexities.json`
        - `epoch_1_perplexities.json`
        - etc.
        
        These should be the output files from `calculate_perplexity.py`.
        """)
        return
    
    # Load epoch data
    with st.spinner("Loading epoch files..."):
        epoch_data, errors = load_epoch_files(uploaded_files)
    
    if errors:
        st.error("❌ Errors loading files:")
        for error in errors:
            st.error(f"  • {error}")
        if not epoch_data:
            return
    
    if not epoch_data:
        st.error("❌ No valid epoch data found")
        return
    
    st.success(f"✅ Loaded {len(epoch_data)} epochs successfully!")
    
    # Display epoch summary
    st.header("📊 Data Summary")
    
    epochs_available = sorted(epoch_data.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Epochs Available", len(epochs_available))
    with col2:
        first_epoch = epochs_available[0]
        num_documents = len(epoch_data[first_epoch])
        st.metric("Documents", num_documents)
    with col3:
        st.metric("Epoch Range", f"{min(epochs_available)} to {max(epochs_available)}")
    
    st.info(f"📅 Available epochs: {epochs_available}")
    
    # Document selection
    st.header("📄 Document Selection")
    
    # Get available text IDs (should be same across epochs)
    first_epoch = epochs_available[0]
    available_text_ids = list(epoch_data[first_epoch].keys())
    
    selected_text_id = st.selectbox(
        "Choose a document to explore:",
        available_text_ids,
        help="Select which document you want to explore"
    )
    
    if selected_text_id:
        # Display document info
        doc_data = epoch_data[first_epoch][selected_text_id]
        doc_length = len(doc_data['text'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document Length", f"{doc_length} characters")
        with col2:
            st.metric("Token Count", len(doc_data['tokens']))
        with col3:
            st.metric("Text ID", selected_text_id)
        
        # Document character display
        st.header("📖 Document Content")
        st.markdown("**Use this to identify which characters/sequences you want to analyze:**")
        
        # Display document with character positions
        doc_text = doc_data['text']
        
        # Create simple position:character format
        char_pairs = []
        for i, char in enumerate(doc_text):
            # Display character (replace special characters with visible symbols)
            display_char = char
            if char == '\n':
                display_char = '↩'
            elif char == '\t':
                display_char = '→'
            elif char == ' ':
                display_char = '·'
            
            char_pairs.append(f"{i}:{display_char}")
        
        # Create readable text with line breaks every 10-15 items
        items_per_line = 12
        formatted_lines = []
        for i in range(0, len(char_pairs), items_per_line):
            line_items = char_pairs[i:i+items_per_line]
            formatted_lines.append(" ".join(line_items))
        
        formatted_text = "\n".join(formatted_lines)
        
        # Display in a text area for easy scrolling and selection
        st.text_area(
            "📋 Document Characters with Positions",
            value=formatted_text,
            height=300,
            help="Format: position:character. Special chars: · = space, ↩ = newline, → = tab"
        )
        
        # Quick navigation helper
        st.markdown("**💡 Tip:** Use the character positions above to choose your start_index and end_index below")
        
        # Sequence selection
        st.header("🎯 Sequence Selection")
        
        # Sequence selection form
        with st.form("sequence_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                start_idx = st.number_input(
                    "Start Index",
                    min_value=0,
                    max_value=doc_length-1,
                    value=0,
                    help="Starting position in the document (see character positions above)"
                )
            with col2:
                end_idx = st.number_input(
                    "End Index",
                    min_value=1,
                    max_value=doc_length-1,
                    value=min(16, doc_length-1),
                    help="Ending position in the document (inclusive - see character positions above)"
                )
            
            # Calculate sequence length (end_idx is inclusive)
            sequence_length = end_idx - start_idx + 1
            
            # Show sequence preview
            if end_idx < doc_length:
                # Use fixed context for preview
                left_context = 50
                right_context = 50
                
                before_seq, sequence_text, after_seq = get_sequence_with_context(
                    epoch_data, selected_text_id, start_idx, end_idx, left_context, right_context
                )
                
                st.subheader("Sequence Preview")
                context_html = f"""
                <div style="font-family: monospace; padding: 15px; background-color: #f5f5f5; border-radius: 5px; margin: 10px 0;">
                    <span style="color: #666;">{before_seq}</span><span style="background-color: #e3f2fd; font-weight: bold; padding: 2px; border: 2px solid #1976d2;">{sequence_text}</span><span style="color: #666;">{after_seq}</span>
                </div>
                """
                st.markdown(context_html, unsafe_allow_html=True)
            
            # Visualization settings
            st.subheader("🎨 Visualization Settings")
            
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
            
            submitted = st.form_submit_button("🔍 Explore Sequence", type="primary")
        
        # Generate visualization
        if submitted:
            if end_idx >= doc_length:
                st.error(f"End index {end_idx} exceeds document length ({doc_length-1})!")
            elif start_idx > end_idx:
                st.error("Start index must be less than or equal to end index!")
            else:
                # Extract sequence data
                seq_data, seq_error = extract_sequence_data(epoch_data, selected_text_id, start_idx, end_idx)
                
                if seq_error:
                    st.error(f"❌ {seq_error}")
                else:
                    # Store visualization data in session state
                    st.session_state.viz_data = seq_data
                    st.session_state.viz_settings = {
                        'use_log_scale': use_log_scale,
                        'alpha_value': alpha_value,
                        'start_color': start_color,
                        'end_color': end_color,
                        'line_width': line_width,
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
                        'marker_size': marker_size,
                        'selected_text_id': selected_text_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'sequence_length': sequence_length
                    }
                    st.session_state.visualization_generated = True

        # Display visualization if it has been generated
        if hasattr(st.session_state, 'visualization_generated') and st.session_state.visualization_generated and hasattr(st.session_state, 'viz_data'):
            viz_data = st.session_state.viz_data
            settings = st.session_state.viz_settings
            
            st.header(f"📈 Sequence Visualization [{settings['start_idx']}:{settings['end_idx']}]")
            
            # Show sequence info
            epochs_in_data = viz_data['epoch_numbers']
            st.info(f"📅 Showing sequence across {len(epochs_in_data)} epochs: {epochs_in_data}")
            
            # Get current color values for the interactive plot
            current_start_color = st.session_state.get('start_color_picker', settings['start_color'])
            current_end_color = st.session_state.get('end_color_picker', settings['end_color'])
            
            # Create and display plot
            fig = plot_perplexity_distribution(
                viz_data, settings['use_log_scale'], settings['alpha_value'], current_start_color, current_end_color,
                settings['graph_width'], settings['graph_height'], settings['marker_size'], settings['legend_position'],
                settings['line_width'], settings['show_grid'], settings['grid_alpha'], settings['font_size'], settings['bg_color']
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
                                viz_data, settings['use_log_scale'], settings['alpha_value'], current_start_color, current_end_color,
                                settings['graph_width_inches'], settings['graph_height_inches'], settings['marker_size'], 
                                dpi=settings['export_dpi'],
                                legend_position=settings['legend_position'].lower().replace(" (inside)", "").replace(" (outside)", ""),
                                line_width=settings['line_width'], show_grid=settings['show_grid'], 
                                grid_alpha=settings['grid_alpha'], font_size=settings['font_size'], bg_color=settings['bg_color']
                            )
                            
                            # Generate Python code
                            python_code = generate_python_code(
                                text_id=settings['selected_text_id'],
                                start_idx=settings['start_idx'],
                                end_idx=settings['end_idx'],
                                start_color=current_start_color,
                                end_color=current_end_color,
                                use_log_scale=settings['use_log_scale'], 
                                alpha=settings['alpha_value'], 
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
                            
                            filename = f"sequence_{settings['selected_text_id']}_{settings['start_idx']}_{settings['end_idx']}.png"
                            
                            st.session_state['png_data'] = png_buffer.getvalue()
                            st.session_state['png_filename'] = filename
                            st.session_state['python_code'] = python_code
                            
                            st.success("✅ PNG and Python code generated!")
                    except Exception as e:
                        st.error(f"Error creating PNG: {str(e)}")
            
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
                    python_filename = f"sequence_{settings['selected_text_id']}_{settings['start_idx']}_{settings['end_idx']}_visualization.py"
                    st.download_button(
                        label="📄 Download Python Code",
                        data=st.session_state['python_code'],
                        file_name=python_filename,
                        mime="text/plain",
                        key="download_python_btn",
                        type="secondary"
                    )
                with col2:
                    st.info("💡 Update the epoch files directory path in the downloaded code and run it to recreate the visualization.")
            
            # Sequence statistics
            st.subheader("📊 Sequence Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sequence Length", f"{settings['sequence_length']} characters")
            with col2:
                st.metric("Available Epochs", len(epochs_in_data))
            with col3:
                all_perps = []
                for perps in viz_data['epoch_perplexities'].values():
                    all_perps.extend(perps)
                if all_perps:
                    st.metric("Perplexity Range", f"{min(all_perps):.1f} - {max(all_perps):.1f}")
            with col4:
                if len(epochs_in_data) >= 2:
                    first_epoch = min(epochs_in_data)
                    last_epoch = max(epochs_in_data)
                    
                    # Calculate perplexity change
                    first_perps = viz_data['epoch_perplexities'][first_epoch]
                    last_perps = viz_data['epoch_perplexities'][last_epoch]
                    
                    first_avg = np.mean(first_perps)
                    last_avg = np.mean(last_perps)
                    change = ((last_avg - first_avg) / first_avg) * 100
                    
                    direction = "↓" if change < 0 else "↑"
                    st.metric("Perplexity Change", f"{direction} {abs(change):.1f}%")
        
        # Add a button to clear selection
        if hasattr(st.session_state, 'visualization_generated') and st.session_state.visualization_generated:
            if st.button("🔄 Clear Visualization", type="secondary"):
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