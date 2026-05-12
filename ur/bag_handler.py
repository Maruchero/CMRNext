from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.reader import read_protobuf_messages


# --- Helper functions (get_topic_info, etc.) remain the same ---
def get_topic_info(bag_file, ros_version="JAZZY") -> List[tuple]:
    """Get topic information from bag file."""
    topics = []

    with open(bag_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        if summary and summary.channels and summary.statistics:
            for channel_id, channel in summary.channels.items():
                msg_count = summary.statistics.channel_message_counts.get(channel_id, 0)
                schema = summary.schemas[channel.schema_id]
                topics.append((channel.topic, schema.name, msg_count))

    return topics


def get_total_message_count(bag_file, ros_version="JAZZY") -> int:
    """Get total message count from bag file."""

    with open(bag_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary and summary.statistics:
            return summary.statistics.message_count

    return 0


def iterate_all_messages(bag_file: str, ros_version="JAZZY"):
    """Generator to iterate through all messages in the mcap file efficiently."""
    
    with open(bag_file, "rb") as f:
        for message in read_protobuf_messages(f):
            yield message.log_time, message.topic, message.proto_msg


def combine_tf_static_messages(tf_messages: List[Any]) -> Optional[Any]:
    """Combine multiple tf_static messages into one composite message."""
    if not tf_messages:
        return None
    if len(tf_messages) == 1:
        return tf_messages[0]

    base_message = tf_messages[0]
    all_transforms = list(getattr(base_message, "transforms", []))
    seen = {(tf.header.frame_id, tf.child_frame_id) for tf in all_transforms}

    for msg in tf_messages[1:]:
        for tf in getattr(msg, "transforms", []):
            key = (tf.header.frame_id, tf.child_frame_id)
            if key not in seen:
                all_transforms.append(tf)
                seen.add(key)

    if hasattr(base_message, "transforms"):
        base_message.transforms = all_transforms
    return base_message


def read_all_messages_optimized(
    bag_file: str,
    topics_to_read: dict,
    topic_message_counts,
    progress_callback=None,
    total_messages=None,
    frame_samples: int = 6,
    ros_version: str = "JAZZY",
) -> dict:
    """Read multiple uniformly sampled frame messages from rosbag."""
    sensor_topics = {t: mt for t, mt in topics_to_read.items() if "tf_static" not in t}
    topic_counts = {t: topic_message_counts.get(t, 0) for t in sensor_topics.keys()}
    sampling_intervals = {t: max(1, c // frame_samples) for t, c in topic_counts.items() if c > 0}

    messages = {}
    tf_static_messages = []
    msg_counters = {topic: 0 for topic in sensor_topics.keys()}
    collected_samples = {topic: [] for topic in sensor_topics.keys()}

    processed_count = 0
    for timestamp, topic, msg_data in iterate_all_messages(bag_file, ros_version):
        processed_count += 1
        if progress_callback and total_messages and processed_count % 100 == 0:
            progress = int((processed_count / total_messages) * 70) + 20
            progress_callback.emit(
                min(progress, 90), f"Sampling frames {processed_count}/{total_messages}..."
            )

        if topic in topics_to_read:
            if "tf_static" in topic:
                tf_static_messages.append(msg_data)
            elif topic in sensor_topics:
                msg_counters[topic] += 1
                interval = sampling_intervals.get(topic, 1)
                if (msg_counters[topic] - 1) % interval == 0 and len(
                    collected_samples[topic]
                ) < frame_samples:
                    collected_samples[topic].append(
                        {
                            "timestamp": timestamp,
                            "data": msg_data,
                            "topic_type": topics_to_read[topic],
                        }
                    )

    messages["frame_samples"] = collected_samples
    if tf_static_messages:
        tf_topic = next((t for t in topics_to_read if "tf_static" in t), None)
        if tf_topic:
            messages[tf_topic] = combine_tf_static_messages(tf_static_messages)

    if progress_callback:
        progress_callback.emit(
            90, f"Collected {sum(len(s) for s in collected_samples.values())} frame samples."
        )
    return messages
