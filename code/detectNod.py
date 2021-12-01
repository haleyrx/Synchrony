import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import argparse
import time

def read_openface_data(filepath):
    '''
    Read in OpenFace outputs stored in csv file

    Parameters:
        filepath (string): path to csv file with OpenFace outputs
    Returns:
        df (dataframe): dataframe with selected columns from OpenFace
    '''

    fields = ['frame', 'timestamp', 'x_33', 'y_33']
    df = pd.read_csv(filepath, skipinitialspace=True, usecols=fields)
    return df


def get_threshold(series, alpha=0.5):
    '''
    Calculate threshold to detect head nods using normalization; a point is considered a peak if it's y-axis
    pixels exceeds the mean + std. dev. * alpha

    Parameters:
        series (pandas series): series with 2D pixels data
        alpha (float): factor to apply to standard deviation
    Returns:
        (float): value in pixels
    '''

    mu = series.mean()
    sigma = series.std()
    return mu + alpha * sigma


def estimate_frequency(signal, fs, plot=False):
    '''
    Estimate the frequency of a signal using real fast fourier transform; frequency is the max frequency over the
    specified portion of signal

    Parameters:
        signal (array): list of signal values
        fs (integer): sampling frequency of signal
        plot (boolean): generate spectrogram for signal
    Returns:
        (float): estimated frequency in Hz
    '''

    yf = rfft(signal)
    xf = rfftfreq(len(signal), 1 / fs)
    yf[0] = 0
    i = np.argmax(abs(yf))

    # plot spectrum with freq
    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xf, abs(yf), linewidth=1.2)
        ax.set_title('nod spectrogram', size=14)

    return np.abs(xf)[i]


def detect_nod(df, peak_indices, gap_threshold):
    '''
    Detect when nodding occurs based on the OpenFace data

    Parameters:
        df (dataframe): dataframe with y-axis movement openface outputs
        peak_indices (list): list of indentified peaks in signal
        gap_threshold (float): threshold cutoff in seconds used to group together peaks as a part of a single nod
    Returns:
        headnods (nested list): list of clusters of nods and the associated indices in df
    '''
    headnods = []
    nod_num = 0

    # time of previous peak
    prev_peak_timestamp = None

    for index, row in df.iterrows():
        if index in peak_indices:
            current_timestamp = df.loc[index, 'timestamp']

            # peak is part of same nod if peak is less than [gap_threshold] seconds away from previous peak
            if prev_peak_timestamp and (current_timestamp - prev_peak_timestamp) < gap_threshold:
                headnods[nod_num].append(index)

            # new head nod detected
            elif prev_peak_timestamp and (current_timestamp - prev_peak_timestamp) >= gap_threshold:
                nod_num += 1
                headnods.append([index])

            elif not prev_peak_timestamp:
                headnods.append([index])

            # update the previous peak timestamp
            prev_peak_timestamp = current_timestamp

    return headnods


def get_statistics(df, signal_column, nod_array, fs):
    '''
    Get dataframe of nod cluster data including frequencies and duration

    Parameters:
        df (dataframe): dataframe with OpenFace outputs
        signal_column (string): column to use as signal to detect nods
        nod_array (array): list of peaks calculated from signal_column
        fs (integer): sampling frequency
    Returns:
        nod_num (integer): number of nod clusters detected
        df_stats (dataframe): nod statistics
    '''
    df_stats = pd.DataFrame(
        columns=['nod_id', 'peak_indices', 'peak_timestamps', 'start_time', 'end_time', 'duration', 'frequency'])

    for i, nod in enumerate(nod_array):
        # calculate frequency of nodding; note that frequency is NaN if nod cluster consists only of one nod
        if len(nod) > 1:
            nod_signal = list(df.loc[nod[0]:nod[-1], signal_column])
            nod_freq = estimate_frequency(nod_signal, fs)
        else:
            nod_freq = None

        # get times of peaks
        peak_timestamps = df['timestamp'].iloc[nod].to_list()

        # calculate duration of nodding
        end_time = peak_timestamps[-1]
        start_time = peak_timestamps[0]
        duration = end_time - start_time

        # add new row to result dataframe
        row = {'nod_id': i, 'peak_indices': nod, 'peak_timestamps': peak_timestamps, 'start_time': start_time,
               'end_time': end_time, 'duration': duration, 'frequency': nod_freq}
        df_stats = df_stats.append(row, ignore_index=True)

    nod_num = len(nod_array)

    return nod_num, df_stats


def main():
    starttime = time.time()

    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='file path to OpenFace csv')
    parser.add_argument('-o', '--output', help='output file path')
    args = parser.parse_args()

    openface_filepath = args.input

    # detect nod heads
    df_raw = read_openface_data(openface_filepath)
    pixel_threshold = get_threshold(df_raw['y_33'], alpha=0.5)
    peak_indices, _ = find_peaks(df_raw['y_33'], height=pixel_threshold)
    nod_list = detect_nod(df_raw, peak_indices=peak_indices, gap_threshold=1.5)
    number_nods, stats = get_statistics(df_raw, signal_column='y_33', nod_array=nod_list, fs=30)

    endtime = time.time()
    total_runtime = endtime - starttime

    # print and save results
    print(f'Total number of nods detected: {number_nods}')
    print(f'Total execution time: {round(total_runtime ,2)} seconds')
    stats.to_csv(args.output, index=False)



if __name__ == "__main__":
    main()