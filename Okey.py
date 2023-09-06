import os
import csv
from bs4 import BeautifulSoup


def parse_html_file(file_path, tag_pattern):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        columns = soup.find_all(tag_pattern)
        parsed_data = []
        for column in columns:
            column_data = []
            for element in column:
                if element.string:
                    column_data.append(element.string.strip())
            parsed_data.append(column_data)
        return parsed_data


def save_to_csv(file_path, data):
    csv_path = os.path.splitext(file_path)[0] + '.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f'Data saved to {csv_path}')


def run_html_parser(tag_pattern):
    html_files = [file for file in os.listdir('.') if file.endswith('.html')]

    if not html_files:
        print('No HTML files found in the current directory.')
        return

    for html_file in html_files:
        try:
            data = parse_html_file(html_file, tag_pattern)
            save_to_csv(html_file, data)
        except Exception as e:
            print(f'Error parsing {html_file}: {str(e)}')


tag_pattern = input('Enter HTML tags: ')

run_html_parser(tag_pattern)
