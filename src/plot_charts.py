
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from skimage.io import imread
from datetime import datetime
import matplotlib.dates as mdates
import sys

def remove_extremes(dates, data):
    """Remove the min and max values and return modified data."""

    # If there are fewer than 3 data points, there's nothing to remove
    if len(data) < 3:
        return dates, data, [], []

    # Find the indices of the min and max values
    min_index = np.argmin(data)
    max_index = np.argmax(data)

    # Get the removed points and corresponding dates
    removed_data = [data[min_index], data[max_index]]
    removed_dates = [dates[min_index], dates[max_index]]

    # Remove the min and max values from the data and dates
    modified_data = [data[i] for i in range(len(data)) if i != min_index and i != max_index]
    modified_dates = [dates[i] for i in range(len(dates)) if i != min_index and i != max_index]

    return modified_dates, modified_data, removed_dates, removed_data


def plot_charts(input_folder, output_filename, title):
    dates = []
    areas = []
    border_lengths = []

    # Read data from images
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.png'):
            # Extract the date from the filename (YYYYMMDD format)
            date_str = filename.split('.')[0]
            if len(date_str) == 8:
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)
            else:
                print(f'Invalid date form in filename: {filename}')
                continue
            
            # Read the image (could be RGB or grayscale)
            img = imread(os.path.join(input_folder, filename))

            # If the binary mask was written as an RGB image
            if img.ndim == 3:
                img = color.rgb2gray(img)
            
            # Threshold to create a binary mask
            mask = img > 0.5

            # Area
            area = np.sum(mask)
            areas.append(area)

            # Border length
            contours = measure.find_contours(mask, 0.5)
            
            # Sum the lengths of the contours
            border_length = 0
            for contour in contours:
                border_length += len(contour)
            
            # Append to the border lengths list
            border_lengths.append(border_length)

    # Remove extremes from both areas and border_lengths, along with their corresponding dates
    trimmed_dates, trimmed_areas, removed_dates_areas, removed_areas = remove_extremes(dates, areas)
    trimmed_dates_border, trimmed_border_lengths, removed_dates_border, removed_border_lengths = remove_extremes(dates, border_lengths)

    # Plotting 2x2 grid
    plt.figure(figsize=(14, 10))

    # 1st Row: Area vs Date
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(dates, areas, marker='o', color='b', linestyle='-', label='Area (pixels)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Area (pixels)')
    ax1.set_title('Area vs Date')
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.tick_params(axis='x', rotation=90)

    # Trend line for Area
    area_trend = np.polyfit(mdates.date2num(dates), areas, 1)
    area_trend_line = np.poly1d(area_trend)
    ax1.plot(dates, area_trend_line(mdates.date2num(dates)), linestyle='--', color='gray', label='Trend Line')

    # 2nd Row: Trimmed Area vs Date
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(trimmed_dates, trimmed_areas, marker='o', color='b', linestyle='-', label='Trimmed Area (pixels)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Area (pixels)')
    ax3.set_title('Area vs Date')
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.tick_params(axis='x', rotation=90)

    # Plot removed points (grayed out)
    ax3.scatter(removed_dates_areas, removed_areas, color='orange', s=50, edgecolors='red', linewidths=2, label='Removed Points')

    # Trend line for Area (Without Extremes)
    area_trend_trimmed = np.polyfit(mdates.date2num(trimmed_dates), trimmed_areas, 1)
    area_trend_line_trimmed = np.poly1d(area_trend_trimmed)
    ax3.plot(trimmed_dates, area_trend_line_trimmed(mdates.date2num(trimmed_dates)), linestyle='--', color='gray', label='Trend Line (Trimmed)')

    # 1st Row: Original Border Length vs Date
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(dates, border_lengths, marker='o', color='r', linestyle='-', label='Border Length (pixels)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Border Length (pixels)')
    ax2.set_title('Border Length vs Date')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.tick_params(axis='x', rotation=90)

    # Trend line for Border Length (Original)
    border_trend = np.polyfit(mdates.date2num(dates), border_lengths, 1)
    border_trend_line = np.poly1d(border_trend)
    ax2.plot(dates, border_trend_line(mdates.date2num(dates)), linestyle='--', color='gray', label='Trend Line')

    # 2nd Row: Trimmed Border Length vs Date
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(trimmed_dates_border, trimmed_border_lengths, marker='o', color='r', linestyle='-', label='Trimmed Border Length (pixels)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Border Length (pixels)')
    ax4.set_title('Border Length vs Date')
    ax4.grid(True)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax4.tick_params(axis='x', rotation=90)

    # Plot removed points (grayed out)
    ax4.scatter(removed_dates_border, removed_border_lengths, color='lime', s=50, edgecolors='darkgreen', linewidths=2, label='Removed Points')


    # Trend line for Border Length (Without Extremes)
    border_trend_trimmed = np.polyfit(mdates.date2num(trimmed_dates_border), trimmed_border_lengths, 1)
    border_trend_line_trimmed = np.poly1d(border_trend_trimmed)
    ax4.plot(trimmed_dates_border, border_trend_line_trimmed(mdates.date2num(trimmed_dates_border)), linestyle='--', color='gray', label='Trend Line (Trimmed)')

    # Add common title above both rows
    plt.suptitle(f'Area and border plots over time by {title} method.', fontsize=16, ha='center')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the title

    # Save the figure
    plt.savefig(output_filename, bbox_inches='tight')
    print(f'Writing {output_filename}')



if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: python plot_charts.py <input_folder> <output_filename> <chart title>')
        print(f'Example: python plot_charts.py /in/dir /out/dir/filename.pdf "Random Forest"')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_filename = sys.argv[2]
    title = sys.argv[3]

    plot_charts(input_folder, output_filename, title)






