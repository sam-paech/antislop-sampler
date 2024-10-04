# Calculates a slop score for a provided text

import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed

def load_and_preprocess_slop_words():
    with open('slop_phrase_prob_adjustments.json', 'r') as f:
        slop_phrases = json.load(f)
    
    phrase_weighting = [1.0 - prob_adjustment for word, prob_adjustment in slop_phrases]
    max_score = max(phrase_weighting)
    scaled_weightings = [score / max_score for score in phrase_weighting]
    n_slop_words = 600
    return {word.lower(): score for (word, _), score in zip(slop_phrases[:n_slop_words], scaled_weightings[:n_slop_words])}

def extract_text_blocks(file_path, compiled_pattern):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    matches = compiled_pattern.findall(content)
    return '\n'.join(matches)

def calculate_slop_score_chunk(args):
    text, slop_words_chunk = args
    return sum(
        score * len(re.findall(r'\b' + re.escape(word) + r'\b', text))
        for word, score in slop_words_chunk.items()
    )

def calculate_and_plot_slop_indices(slop_indices):
    if not slop_indices:
        print("No slop indices to plot.")
        return []
    
    # Sort the indices in descending order
    sorted_indices = sorted(slop_indices.items(), key=lambda x: x[1], reverse=True)
    models, indices = zip(*sorted_indices) if sorted_indices else ([], [])
    
    # Set the style for better aesthetics
    plt.style.use('seaborn-darkgrid')  # You can choose other styles like 'ggplot', 'fivethirtyeight', etc.
    
    plt.figure(figsize=(12, 18))
    
    # Create a horizontal bar chart
    bars = plt.barh(models, indices, color=plt.cm.viridis(range(len(indices))))
    
    plt.title('Slop Index by Model', fontsize=16, weight='bold', pad=15)
    plt.xlabel('Slop Index', fontsize=14, labelpad=10)
    plt.ylabel('Model', fontsize=14, labelpad=10)
    
    # Invert y-axis to have the highest slop index on top
    plt.gca().invert_yaxis()
    
    # Add value labels to each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(indices)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center', fontsize=12)
    
    # Customize x-axis ticks
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save the figure with higher resolution
    plt.savefig('slop_index_chart.png', dpi=300)
    plt.show()
    plt.close()
    
    return sorted_indices

def split_into_chunks(slop_words, num_chunks):
    slop_words_items = list(slop_words.items())
    chunk_size = len(slop_words_items) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    return [dict(slop_words_items[i:i + chunk_size]) for i in range(0, len(slop_words_items), chunk_size)]


# Call this to function to calculate a slop score.
# This is the way it's calculated for the eqbench creative writing leaderboard.
def calculate_slop_index(extracted_text):    
    slop_words = load_and_preprocess_slop_words()
    
    num_chunks = 12 #mp.cpu_count()
    slop_words_chunks = split_into_chunks(slop_words, num_chunks)
    
    if not extracted_text:
        slop_index = 0.0
    else:
        # Parallelize the calculation using joblib
        slop_scores = Parallel(n_jobs=num_chunks)(delayed(calculate_slop_score_chunk)((extracted_text, chunk)) for chunk in slop_words_chunks)
        
        slop_score = sum(slop_scores)
        total_words = len(extracted_text.split())
        slop_index = (slop_score / total_words) * 1000 if total_words > 0 else 0
    return slop_index

