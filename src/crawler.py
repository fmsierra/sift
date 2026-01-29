import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os
import sys
import time
import pandas as pd

def get_leidsa_results(date_obj):
    """
    Crawls the loteriasdominicanas.com website for Leidsa results on a specific date object.
    Extracts 'Loto - Super Loto Más' only on Wednesdays and Sundays.
    Returns: 
        (status, data)
        status: 'skipped' (wrong day), 'error' (network/parse), 'not_found' (game missing), 'success'
        data: extracted dict or None
    """
    date_str = date_obj.strftime("%d-%m-%Y")

    # 1. Check Day of Week
    # Monday=0, Sunday=6
    weekday = date_obj.weekday()
    
    # Wednesday = 2, Sunday = 6
    if weekday not in [2, 6]:
        return 'skipped', None

    url = f"https://loteriasdominicanas.com/leidsa?date={date_str}"
    print(f"Fetching: {url}")

    try:
        # Note: In a real environment, you might need headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error requesting page for {date_str}: {e}")
        # Return error but don't crash, we want to try the next date
        return 'error', None

    soup = BeautifulSoup(response.content, 'html.parser')

    # 3. Extract Data
    # Searching for the game block containing "Loto - Super Loto Más"
    target_game_name = "Loto - Super Loto Más"
    
    results = {}
    
    game_found = False
    game_blocks = soup.find_all('div', class_='game-block')
    
    for block in game_blocks:
        title_tag = block.find('a', class_='game-title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            if target_game_name in title_text:
                game_found = True
                
                # Extract Scores
                scores_div = block.find('div', class_='game-scores')
                if not scores_div:
                    print(f"Warning: Scores container not found for {target_game_name} on {date_str}")
                    continue
                    
                score_spans = scores_div.find_all('span', class_='score')
                
                numbers = []
                special_numbers = []
                
                for span in score_spans:
                    val = span.get_text(strip=True)
                    classes = span.get('class', [])
                    
                    # Logic based on class names seen in the HTML snippet
                    # Handles flexible number of balls (6, 8, or any number)
                    if 'special1' in classes:
                        special_numbers.append({'value': val, 'type': 'special1'}) # Loto Más / Super Loto?
                    elif 'special2' in classes:
                        special_numbers.append({'value': val, 'type': 'special2'})
                    else:
                        numbers.append(val)
                
                total_balls = len(numbers) + len(special_numbers)
                
                # Handling the user's specific request about ball counts < 8
                if total_balls < 8:
                     pass # Implicitly handled by just extracting what is there.

                results = {
                    "date": date_str,
                    "day_of_week": date_obj.strftime('%A'),
                    "game": target_game_name,
                    "winning_numbers": numbers,
                    "special_numbers": special_numbers
                }
                break
    
    if not game_found:
        print(f"Info: Game '{target_game_name}' not found on page for {date_str}.")
        return 'not_found', None
        
    return 'success', results

def save_results(data, output_dir="data/raw"):
    if not data:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename based on date
    filename = f"leidsa_loto_{data['date']}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def export_to_csv(all_data, output_path="data/raw/all_draws.csv"):
    if not all_data:
        print("No data to export.")
        return

    flattened_data = []
    
    for item in all_data:
        row = {
            "date": item['date'],
            "day_of_week": item['day_of_week'],
            "game": item['game']
        }
        
        # Add winning numbers as separate columns
        for idx, val in enumerate(item['winning_numbers']):
            row[f"winning_{idx+1}"] = val
            
        # Add special numbers as separate columns
        for idx, s in enumerate(item.get('special_numbers', [])):
             row[f"special_{idx+1}"] = s['value']

        flattened_data.append(row)
        
    df = pd.DataFrame(flattened_data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"CSV Export successful: {output_path}")

def run_crawler_backwards(start_date_str, end_year=2006):
    try:
        current_date = datetime.strptime(start_date_str, "%d-%m-%Y")
    except ValueError:
        print(f"Error: Invalid start date format '{start_date_str}'. Please use DD-MM-YYYY.")
        return

    # Heuristic: If we miss 12 valid days (Wed/Sun) in a row, assume game discontinued
    MAX_CONSECUTIVE_MISSES = 12 
    consecutive_not_found = 0

    print(f"Starting crawl from {start_date_str} backwards to {end_year}...")
    
    all_results = []
    processed_count = 0

    while current_date.year >= end_year:
        status, data = get_leidsa_results(current_date)
        
        if status == 'success':
            save_results(data)
            all_results.append(data)
            
            consecutive_not_found = 0
            processed_count += 1
            
            # Be polite to the server
            time.sleep(0.5) 
            
        elif status == 'not_found':
            consecutive_not_found += 1
            if consecutive_not_found >= MAX_CONSECUTIVE_MISSES:
                print(f"Stopping: Game has not been found for {consecutive_not_found} consecutive game days (Wed/Sun).")
                break
        
        elif status == 'skipped':
            # Not a Wed/Sun, just move on
            pass
            
        elif status == 'error':
            pass

        # Decrement day
        current_date -= timedelta(days=1)

    print(f"Crawl complete. Processed {processed_count} records.")
    
    # Export to CSV at the end
    print("Exporting to CSV...")
    export_to_csv(all_results)

if __name__ == "__main__":
    # Default date from prompt if not provided
    target_date = "28-01-2026"
    
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
        
    run_crawler_backwards(target_date)
