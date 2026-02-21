"""
Streaming Data Pipeline - Simulated Real-Time Processing
=========================================================
Demonstrates real-time data streaming concepts with:
- Producer/consumer pattern using threading
- Synthetic IoT/sensor data generation
- Sliding window and tumbling window aggregations
- Running statistics and anomaly detection
- Event buffering and batch processing
- Pipeline metrics and monitoring
"""

import threading
import queue
import time
import random
import math
import json
import csv
import os
import sys
import logging
import tempfile
import statistics
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("StreamingPipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PIPELINE_CONFIG = {
    "pipeline_name": "iot_sensor_stream",
    "producer": {
        "num_sensors": 5,
        "events_per_second": 10,
        "total_events": 200,
        "anomaly_probability": 0.05,
    },
    "consumer": {
        "batch_size": 10,
        "batch_timeout_sec": 2.0,
    },
    "windows": {
        "tumbling_window_sec": 5,
        "sliding_window_sec": 10,
        "sliding_step_sec": 2,
    },
    "anomaly": {
        "z_score_threshold": 2.5,
        "min_samples": 10,
    },
    "output_dir": None,  # will be set at runtime
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SensorEvent:
    """Represents a single sensor reading."""
    event_id: str
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: float  # epoch seconds
    location: str
    is_anomaly: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


@dataclass
class WindowResult:
    """Result of a windowed aggregation."""
    window_type: str
    window_start: float
    window_end: float
    sensor_id: str
    metric: str
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    anomaly_count: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window_start_iso"] = datetime.fromtimestamp(self.window_start).isoformat()
        d["window_end_iso"] = datetime.fromtimestamp(self.window_end).isoformat()
        return d


@dataclass
class PipelineMetrics:
    """Tracks pipeline performance metrics."""
    events_produced: int = 0
    events_consumed: int = 0
    events_dropped: int = 0
    batches_processed: int = 0
    anomalies_detected: int = 0
    windows_computed: int = 0
    processing_errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.events_consumed == 0:
            return 0.0
        return self.total_latency_ms / self.events_consumed

    @property
    def throughput_eps(self) -> float:
        duration = self.end_time - self.start_time
        if duration == 0:
            return 0.0
        return self.events_consumed / duration

    def summary(self) -> str:
        duration = self.end_time - self.start_time
        lines = [
            "\n=== Pipeline Metrics ===",
            f"  Duration           : {duration:.2f}s",
            f"  Events produced    : {self.events_produced}",
            f"  Events consumed    : {self.events_consumed}",
            f"  Events dropped     : {self.events_dropped}",
            f"  Batches processed  : {self.batches_processed}",
            f"  Anomalies detected : {self.anomalies_detected}",
            f"  Windows computed   : {self.windows_computed}",
            f"  Processing errors  : {self.processing_errors}",
            f"  Avg latency        : {self.avg_latency_ms:.2f} ms",
            f"  Max latency        : {self.max_latency_ms:.2f} ms",
            f"  Throughput         : {self.throughput_eps:.1f} events/sec",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sensor Data Producer
# ---------------------------------------------------------------------------
class SensorDataProducer:
    """Generates synthetic IoT sensor data events."""

    SENSOR_PROFILES = {
        "temperature": {"base": 22.0, "noise": 3.0, "unit": "celsius", "drift": 0.01},
        "humidity": {"base": 55.0, "noise": 8.0, "unit": "percent", "drift": 0.005},
        "pressure": {"base": 1013.25, "noise": 5.0, "unit": "hPa", "drift": 0.002},
        "light": {"base": 500.0, "noise": 100.0, "unit": "lux", "drift": 0.0},
        "co2": {"base": 400.0, "noise": 50.0, "unit": "ppm", "drift": 0.02},
    }

    LOCATIONS = ["Building-A/Floor-1", "Building-A/Floor-2", "Building-B/Floor-1",
                 "Building-B/Floor-2", "Warehouse/Zone-1"]

    def __init__(self, event_queue: queue.Queue, config: dict, metrics: PipelineMetrics):
        self.queue = event_queue
        self.config = config["producer"]
        self.metrics = metrics
        self.rng = random.Random(42)
        self.running = False
        self.sensors = self._init_sensors()
        self.event_counter = 0

    def _init_sensors(self) -> list[dict]:
        sensors = []
        sensor_types = list(self.SENSOR_PROFILES.keys())
        for i in range(self.config["num_sensors"]):
            stype = sensor_types[i % len(sensor_types)]
            profile = self.SENSOR_PROFILES[stype]
            sensors.append({
                "sensor_id": f"SENSOR-{i+1:03d}",
                "sensor_type": stype,
                "base_value": profile["base"],
                "noise": profile["noise"],
                "unit": profile["unit"],
                "drift": profile["drift"],
                "location": self.LOCATIONS[i % len(self.LOCATIONS)],
                "step": 0,
            })
        return sensors

    def _generate_value(self, sensor: dict) -> tuple[float, bool]:
        """Generate a sensor reading with optional anomaly."""
        step = sensor["step"]
        sensor["step"] += 1

        # Base value with drift and sinusoidal pattern
        base = sensor["base_value"] + sensor["drift"] * step
        seasonal = math.sin(step * 0.1) * sensor["noise"] * 0.3
        noise = self.rng.gauss(0, sensor["noise"] * 0.2)
        value = base + seasonal + noise

        # Inject anomaly
        is_anomaly = False
        if self.rng.random() < self.config["anomaly_probability"]:
            spike = self.rng.choice([-1, 1]) * sensor["noise"] * self.rng.uniform(3, 6)
            value += spike
            is_anomaly = True

        return round(value, 3), is_anomaly

    def produce_event(self) -> SensorEvent:
        sensor = self.rng.choice(self.sensors)
        value, is_anomaly = self._generate_value(sensor)
        self.event_counter += 1
        return SensorEvent(
            event_id=f"EVT-{self.event_counter:06d}",
            sensor_id=sensor["sensor_id"],
            sensor_type=sensor["sensor_type"],
            value=value,
            unit=sensor["unit"],
            timestamp=time.time(),
            location=sensor["location"],
            is_anomaly=is_anomaly,
            metadata={"producer_thread": threading.current_thread().name},
        )

    def run(self):
        """Main producer loop."""
        self.running = True
        total = self.config["total_events"]
        eps = self.config["events_per_second"]
        delay = 1.0 / eps if eps > 0 else 0.1
        logger.info(f"Producer started: generating {total} events at ~{eps} eps")

        for i in range(total):
            if not self.running:
                break
            try:
                event = self.produce_event()
                self.queue.put(event, timeout=5)
                self.metrics.events_produced += 1
            except queue.Full:
                self.metrics.events_dropped += 1
                logger.warning("Event dropped: queue full")
            time.sleep(delay)

        self.running = False
        # Sentinel to signal consumer
        self.queue.put(None)
        logger.info(f"Producer finished: {self.metrics.events_produced} events produced")

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Running Statistics Tracker
# ---------------------------------------------------------------------------
class RunningStatistics:
    """Maintains running statistics using Welford's online algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.values: deque = deque(maxlen=1000)  # keep recent for percentiles

    def update(self, value: float):
        self.n += 1
        self.values.append(value)
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        if self.std == 0 or self.n < 2:
            return 0.0
        return (value - self.mean) / self.std


# ---------------------------------------------------------------------------
# Window Aggregators
# ---------------------------------------------------------------------------
class TumblingWindow:
    """Fixed-size, non-overlapping time windows."""

    def __init__(self, window_size_sec: float):
        self.window_size = window_size_sec
        self.current_window_start: Optional[float] = None
        self.buffers: dict[str, list[SensorEvent]] = defaultdict(list)

    def add_event(self, event: SensorEvent) -> Optional[dict[str, list[SensorEvent]]]:
        """Add event, return completed windows if boundary crossed."""
        if self.current_window_start is None:
            self.current_window_start = event.timestamp

        window_end = self.current_window_start + self.window_size
        if event.timestamp >= window_end:
            # Window completed, return buffered events
            completed = dict(self.buffers)
            completed_start = self.current_window_start
            self.buffers = defaultdict(list)
            self.current_window_start = event.timestamp
            self.buffers[event.sensor_id].append(event)
            return {
                "events": completed,
                "window_start": completed_start,
                "window_end": window_end,
            }

        self.buffers[event.sensor_id].append(event)
        return None


class SlidingWindow:
    """Overlapping time windows with configurable size and step."""

    def __init__(self, window_size_sec: float, step_sec: float):
        self.window_size = window_size_sec
        self.step = step_sec
        self.events: deque[SensorEvent] = deque()
        self.last_emit: Optional[float] = None

    def add_event(self, event: SensorEvent) -> Optional[dict]:
        self.events.append(event)

        # Expire old events
        cutoff = event.timestamp - self.window_size
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

        # Check if we should emit
        if self.last_emit is None:
            self.last_emit = event.timestamp
            return None

        if event.timestamp - self.last_emit >= self.step:
            self.last_emit = event.timestamp
            # Group by sensor
            grouped: dict[str, list[SensorEvent]] = defaultdict(list)
            for e in self.events:
                grouped[e.sensor_id].append(e)
            return {
                "events": dict(grouped),
                "window_start": event.timestamp - self.window_size,
                "window_end": event.timestamp,
            }
        return None


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Detects anomalies using z-score on running statistics per sensor."""

    def __init__(self, z_threshold: float = 2.5, min_samples: int = 10):
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.sensor_stats: dict[str, RunningStatistics] = defaultdict(RunningStatistics)

    def check(self, event: SensorEvent) -> tuple[bool, float]:
        stats = self.sensor_stats[event.sensor_id]
        z = stats.z_score(event.value)
        is_anomaly = abs(z) > self.z_threshold and stats.n >= self.min_samples
        stats.update(event.value)
        return is_anomaly, z

    def get_sensor_summary(self) -> dict:
        return {
            sid: {
                "count": s.n,
                "mean": round(s.mean, 3),
                "std": round(s.std, 3),
                "min": round(s.min_val, 3) if s.min_val != float("inf") else None,
                "max": round(s.max_val, 3) if s.max_val != float("-inf") else None,
            }
            for sid, s in self.sensor_stats.items()
        }


# ---------------------------------------------------------------------------
# Event Consumer / Processor
# ---------------------------------------------------------------------------
class EventConsumer:
    """Consumes and processes sensor events with windowed aggregations."""

    def __init__(self, event_queue: queue.Queue, config: dict, metrics: PipelineMetrics):
        self.queue = event_queue
        self.config = config
        self.metrics = metrics
        self.running = False

        # Windows
        self.tumbling = TumblingWindow(config["windows"]["tumbling_window_sec"])
        self.sliding = SlidingWindow(
            config["windows"]["sliding_window_sec"],
            config["windows"]["sliding_step_sec"],
        )

        # Anomaly detection
        self.anomaly_detector = AnomalyDetector(
            z_threshold=config["anomaly"]["z_score_threshold"],
            min_samples=config["anomaly"]["min_samples"],
        )

        # Batch buffer
        self.batch_buffer: list[SensorEvent] = []
        self.batch_size = config["consumer"]["batch_size"]
        self.batch_timeout = config["consumer"]["batch_timeout_sec"]
        self.last_batch_time = time.time()

        # Results
        self.all_events: list[dict] = []
        self.window_results: list[WindowResult] = []
        self.anomalies: list[dict] = []

    def _compute_window_stats(self, window_data: dict, window_type: str) -> list[WindowResult]:
        results = []
        events_by_sensor = window_data["events"]
        for sensor_id, events in events_by_sensor.items():
            if not events:
                continue
            values = [e.value for e in events]
            anomaly_count = sum(1 for e in events if e.is_anomaly)
            result = WindowResult(
                window_type=window_type,
                window_start=window_data["window_start"],
                window_end=window_data["window_end"],
                sensor_id=sensor_id,
                metric=events[0].sensor_type,
                count=len(values),
                mean=round(statistics.mean(values), 3),
                std=round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
                min_val=round(min(values), 3),
                max_val=round(max(values), 3),
                anomaly_count=anomaly_count,
            )
            results.append(result)
            self.metrics.windows_computed += 1
        return results

    def _process_event(self, event: SensorEvent):
        """Process a single event through the pipeline."""
        receive_time = time.time()
        latency_ms = (receive_time - event.timestamp) * 1000
        self.metrics.total_latency_ms += latency_ms
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)

        # Anomaly detection
        detected, z_score = self.anomaly_detector.check(event)
        if detected:
            event.is_anomaly = True
            self.metrics.anomalies_detected += 1
            anomaly_record = {
                "event_id": event.event_id,
                "sensor_id": event.sensor_id,
                "value": event.value,
                "z_score": round(z_score, 3),
                "timestamp": event.timestamp,
            }
            self.anomalies.append(anomaly_record)
            logger.warning(
                f"ANOMALY: {event.sensor_id} value={event.value:.3f} "
                f"z={z_score:.2f}"
            )

        # Tumbling window
        tumbling_result = self.tumbling.add_event(event)
        if tumbling_result:
            results = self._compute_window_stats(tumbling_result, "tumbling")
            self.window_results.extend(results)

        # Sliding window
        sliding_result = self.sliding.add_event(event)
        if sliding_result:
            results = self._compute_window_stats(sliding_result, "sliding")
            self.window_results.extend(results)

        self.all_events.append(event.to_dict())
        self.metrics.events_consumed += 1

    def _process_batch(self, batch: list[SensorEvent]):
        """Process a batch of events."""
        for event in batch:
            try:
                self._process_event(event)
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                self.metrics.processing_errors += 1
        self.metrics.batches_processed += 1

    def run(self):
        """Main consumer loop with batch processing."""
        self.running = True
        logger.info("Consumer started")

        while self.running:
            try:
                event = self.queue.get(timeout=self.batch_timeout)
                if event is None:
                    # Sentinel - process remaining batch
                    if self.batch_buffer:
                        self._process_batch(self.batch_buffer)
                        self.batch_buffer = []
                    break

                self.batch_buffer.append(event)

                # Flush batch if size reached
                if len(self.batch_buffer) >= self.batch_size:
                    self._process_batch(self.batch_buffer)
                    self.batch_buffer = []
                    self.last_batch_time = time.time()

            except queue.Empty:
                # Timeout - flush partial batch
                if self.batch_buffer:
                    self._process_batch(self.batch_buffer)
                    self.batch_buffer = []
                    self.last_batch_time = time.time()

        self.running = False
        logger.info(f"Consumer finished: {self.metrics.events_consumed} events processed")


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class StreamingPipeline:
    """Orchestrates the streaming pipeline with producer and consumer threads."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or PIPELINE_CONFIG
        self.output_dir = self.config.get("output_dir") or tempfile.mkdtemp(prefix="stream_")
        self.config["output_dir"] = self.output_dir
        self.metrics = PipelineMetrics()
        self.event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.producer = SensorDataProducer(self.event_queue, self.config, self.metrics)
        self.consumer = EventConsumer(self.event_queue, self.config, self.metrics)

    def run(self) -> dict:
        """Execute the streaming pipeline."""
        logger.info("=" * 60)
        logger.info(f"Starting Streaming Pipeline: {self.config['pipeline_name']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        self.metrics.start_time = time.time()

        # Start threads
        producer_thread = threading.Thread(
            target=self.producer.run, name="producer-thread", daemon=True
        )
        consumer_thread = threading.Thread(
            target=self.consumer.run, name="consumer-thread", daemon=True
        )

        consumer_thread.start()
        producer_thread.start()

        # Wait for completion
        producer_thread.join()
        consumer_thread.join()

        self.metrics.end_time = time.time()

        # Save results
        results = self._save_results()
        return results

    def _save_results(self) -> dict:
        """Save pipeline results to files."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save all events
        events_path = os.path.join(self.output_dir, "events.json")
        with open(events_path, "w") as f:
            json.dump(self.consumer.all_events, f, indent=2, default=str)
        logger.info(f"Saved {len(self.consumer.all_events)} events to {events_path}")

        # Save window results
        windows_path = os.path.join(self.output_dir, "window_results.csv")
        if self.consumer.window_results:
            with open(windows_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.consumer.window_results[0].to_dict().keys()
                )
                writer.writeheader()
                for wr in self.consumer.window_results:
                    writer.writerow(wr.to_dict())
            logger.info(
                f"Saved {len(self.consumer.window_results)} window results "
                f"to {windows_path}"
            )

        # Save anomalies
        anomalies_path = os.path.join(self.output_dir, "anomalies.json")
        with open(anomalies_path, "w") as f:
            json.dump(self.consumer.anomalies, f, indent=2, default=str)
        logger.info(f"Saved {len(self.consumer.anomalies)} anomalies to {anomalies_path}")

        # Save sensor summary
        sensor_summary = self.consumer.anomaly_detector.get_sensor_summary()
        summary_path = os.path.join(self.output_dir, "sensor_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sensor_summary, f, indent=2)
        logger.info(f"Saved sensor summary to {summary_path}")

        return {
            "events_file": events_path,
            "windows_file": windows_path,
            "anomalies_file": anomalies_path,
            "summary_file": summary_path,
            "sensor_summary": sensor_summary,
            "metrics": self.metrics,
        }

    def print_report(self, results: dict):
        """Print pipeline execution report."""
        print("\n" + "=" * 70)
        print("STREAMING PIPELINE EXECUTION REPORT")
        print("=" * 70)
        print(f"Pipeline      : {self.config['pipeline_name']}")
        print(f"Output Dir    : {self.output_dir}")
        print(self.metrics.summary())

        print("\n--- Sensor Statistics ---")
        for sid, stats in results["sensor_summary"].items():
            print(
                f"  {sid}: mean={stats['mean']}, std={stats['std']}, "
                f"range=[{stats['min']}, {stats['max']}], n={stats['count']}"
            )

        print(f"\n--- Window Aggregations ---")
        tumbling_count = sum(
            1 for w in self.consumer.window_results if w.window_type == "tumbling"
        )
        sliding_count = sum(
            1 for w in self.consumer.window_results if w.window_type == "sliding"
        )
        print(f"  Tumbling windows: {tumbling_count}")
        print(f"  Sliding windows : {sliding_count}")

        if self.consumer.window_results:
            print("\n  Sample window results (last 5):")
            for wr in self.consumer.window_results[-5:]:
                print(
                    f"    [{wr.window_type}] {wr.sensor_id}: "
                    f"mean={wr.mean}, std={wr.std}, n={wr.count}"
                )

        print(f"\n--- Anomalies Detected ---")
        print(f"  Total anomalies: {len(self.consumer.anomalies)}")
        if self.consumer.anomalies:
            print("  Recent anomalies:")
            for a in self.consumer.anomalies[-5:]:
                print(
                    f"    {a['sensor_id']}: value={a['value']}, z_score={a['z_score']}"
                )

        print(f"\n--- Output Files ---")
        print(f"  Events   : {results['events_file']}")
        print(f"  Windows  : {results['windows_file']}")
        print(f"  Anomalies: {results['anomalies_file']}")
        print(f"  Summary  : {results['summary_file']}")

        print("\n" + "=" * 70)
        print("Streaming pipeline completed successfully!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Streaming Data Pipeline - Data Engineering Portfolio Project")
    print("-" * 55)

    pipeline = StreamingPipeline(config=PIPELINE_CONFIG)
    results = pipeline.run()
    pipeline.print_report(results)


if __name__ == "__main__":
    main()
