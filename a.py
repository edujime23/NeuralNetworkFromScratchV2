import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def plot_data(data_queue):
    """Function to be run in a separate process for plotting."""
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_title("Live Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    plt.ion()  # Turn on interactive mode for live updates if needed

    while True:
        if not data_queue.empty():
            data = data_queue.get()
            if data is None:  # Sentinel value to signal termination
                break
            x, y = data
            line.set_data(x, y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # Small pause to allow GUI to update
        else:
            time.sleep(0.05)  # Prevent busy-waiting

    plt.close(fig)  # Close the figure when done


if __name__ == "__main__":
    data_queue = mp.Queue()
    plot_process = mp.Process(target=plot_data, args=(data_queue,))
    plot_process.start()  # Start the separate plotting process

    # Simulate data generation in the main process
    for i in range(100):
        x_data = list(range(i + 1))
        y_data = [val**2 for val in x_data]
        data_queue.put((x_data, y_data))
        time.sleep(0.1)

    data_queue.put(None)  # Send a sentinel value to stop the plotting process
    plot_process.join()  # Wait for the plotting process to finish
    print("Main process finished.")
