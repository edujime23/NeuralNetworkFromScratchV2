import contextlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
from threading import Thread, Lock, Event
from collections import defaultdict
import time
from network.plugins.base.plugin import Plugin, PluginContext, PluginPriority
from network.plugins.model.hooks import ModelHookPoints
from network.models import Model


class LivePlotPlugin(Plugin):
    """
    Simple live plot that appends new data points and displays complete history.
    Clean, straightforward implementation for monitoring training metrics.
    """

    def __init__(
        self,
        metrics: str | list[str] | None = None,
        name: str = "live_plot",
        priority: int = PluginPriority.LOW,
        update_interval: float = 0.1,
        max_points: int = 1000,
        show_plot: bool = True,
        save_plot: bool = False,
        line_width: float = 1.5,
        alpha: float = 0.8,
    ):
        super().__init__(name=name, priority=priority)

        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.update_interval = update_interval
        self.max_points = max_points
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.line_width = line_width
        self.alpha = alpha

        # Data storage
        self._data_lock = Lock()
        self._metric_data = defaultdict(list)
        self._epoch_data = []  # Store actual epoch numbers
        self._epoch_counter = 0

        # Plot components
        self._fig = None
        self._ax = None
        self._lines = {}
        self._animation = None
        self._plot_thread = None
        self._stop_event = Event()
        self._plot_initialized = False

        # Colors for different metrics
        self._colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self._color_index = 0

    def get_hook_points(self) -> list[ModelHookPoints]:
        return [ModelHookPoints.POST_EPOCH.value, ModelHookPoints.POST_BATCH.value]

    def _validate_host(self, host: Model) -> None:
        if not isinstance(host, Model):
            raise ValueError("LivePlotPlugin can only attach to Model hosts")

    def _initialize_plot(self) -> None:
        """Initialize matplotlib plot"""
        if self.show_plot:
            matplotlib.use("TkAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        self._fig, self._ax = plt.subplots(figsize=(12, 8))
        self._fig.patch.set_facecolor("white")

        self._ax.set_title("Training Metrics", fontsize=14, fontweight="bold")
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Value")
        self._ax.grid(True, alpha=0.3)

        if self.show_plot:
            plt.show(block=False)
            self._fig.canvas.draw()

        self._plot_initialized = True

    def _get_next_color(self) -> tuple:
        """Get next color for new metric"""
        color = self._colors[self._color_index % len(self._colors)]
        self._color_index += 1
        return color

    def _add_data_point(self, metric_name: str, value: float) -> None:
        """Add new data point"""
        with self._data_lock:
            self._metric_data[metric_name].append(value)

            # Limit data points to prevent memory issues
            if len(self._metric_data[metric_name]) > self.max_points:
                self._metric_data[metric_name] = self._metric_data[metric_name][
                    -self.max_points :
                ]

    def _update_plot(self, frame) -> list:
        """Update plot with new data"""
        if not self._plot_initialized:
            return []

        updated_lines = []

        with self._data_lock:
            # Use actual epoch numbers for x-axis
            current_epochs = self._epoch_data.copy()

            if not current_epochs:
                return []

            for metric_name, y_data in self._metric_data.items():
                if not y_data:
                    continue

                # Ensure x and y data have same length
                x_data = current_epochs[-len(y_data) :]

                if metric_name not in self._lines:
                    # Create new line
                    color = self._get_next_color()
                    (line,) = self._ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        alpha=self.alpha,
                        label=metric_name,
                        marker="o",
                        markersize=3,
                    )
                    self._lines[metric_name] = line

                    # Update legend
                    self._ax.legend(loc="upper left", framealpha=0.8)
                else:
                    # Update existing line
                    self._lines[metric_name].set_data(x_data, y_data)

                updated_lines.append(self._lines[metric_name])

            # Auto-scale axes with proper epoch numbers
            if updated_lines and current_epochs:
                self._ax.relim()
                self._ax.autoscale_view()

                # Set x-axis limits to show scrolling effect
                if len(current_epochs) > 0:
                    x_min, x_max = min(current_epochs), max(current_epochs)
                    # Add some padding for better visualization
                    x_range = max(1, x_max - x_min)
                    self._ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)

        return updated_lines

    def _plot_worker(self) -> None:
        """Background thread for plot management"""
        try:
            self._initialize_plot()

            # Start animation
            self._animation = FuncAnimation(
                self._fig,
                self._update_plot,
                interval=int(self.update_interval * 1000),
                blit=False,
                cache_frame_data=False,
            )

            # Keep thread alive
            while not self._stop_event.is_set():
                if self.show_plot:
                    self._fig.canvas.flush_events()

                if self.save_plot:
                    with contextlib.suppress(Exception):
                        self._fig.savefig(
                            "training_metrics.png", dpi=100, bbox_inches="tight"
                        )

                time.sleep(0.1)

        except Exception as e:
            print(f"[LivePlotPlugin] Plot worker error: {e}")

    def _on_attach(self, host: Model) -> None:
        """Start plotting system"""
        self._stop_event.clear()
        self._plot_thread = Thread(
            target=self._plot_worker, daemon=True, name="LivePlotWorker"
        )
        self._plot_thread.start()

    def _on_detach(self, host: Model) -> None:
        """Clean up resources"""
        self._stop_event.set()

        if self._animation:
            self._animation.event_source.stop()

        if self._plot_thread and self._plot_thread.is_alive():
            self._plot_thread.join(timeout=3.0)

        if self._fig:
            plt.close(self._fig)

        # Clear data
        with self._data_lock:
            self._metric_data.clear()
            self._epoch_data.clear()

    def on_post_epoch(self, context: PluginContext) -> None:
        """Handle epoch updates"""
        self._epoch_counter += 1
        self._process_context_update(context, is_epoch=True)

    def on_post_batch(self, context: PluginContext) -> None:
        """Handle batch updates"""
        self._process_context_update(context, is_epoch=False)

    def _process_context_update(
        self, context: PluginContext, is_epoch: bool = False
    ) -> None:
        """Process update from training context"""
        logs = context.metadata.get("logs", {})

        # Filter metrics if specified
        if self.metrics:
            filtered_logs = {k: v for k, v in logs.items() if k in self.metrics}
        else:
            filtered_logs = logs

        # Add data points
        for key, value in filtered_logs.items():
            try:
                float_value = float(value)
                self._add_data_point(key, float_value)
            except (ValueError, TypeError):
                continue

        # Add current epoch number for every data point (batch or epoch)
        with self._data_lock:
            current_epoch = self._epoch_counter if self._epoch_counter > 0 else 1
            self._epoch_data.append(current_epoch)

            # Limit epoch data length
            if len(self._epoch_data) > self.max_points:
                self._epoch_data = self._epoch_data[-self.max_points :]

    def get_stats(self) -> dict:
        """Get statistics"""
        with self._data_lock:
            return {
                "total_epochs": self._epoch_counter,
                "metrics_tracked": list(self._metric_data.keys()),
                "data_points": {
                    name: len(data) for name, data in self._metric_data.items()
                },
            }
