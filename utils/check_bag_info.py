#!/usr/bin/env python3
"""
Check ROS bag file contents and display topic information.

Usage:
    python check_bag_info.py <path_to_bag_file>

Example:
    python check_bag_info.py "D:\\DESKTOP\\push_block_000.bag"
"""

from rosbags.highlevel import AnyReader
from pathlib import Path
import sys
import os
from collections import defaultdict
import numpy as np


def format_bytes(bytes_size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_duration(nanoseconds):
    """Convert nanoseconds to human-readable duration."""
    seconds = nanoseconds / 1e9
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def format_message_sample(msg, max_depth=2, current_depth=0, indent="  "):
    """
    Format a message for display with limited depth.

    Args:
        msg: Message object
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        indent: Indentation string
    """
    if current_depth >= max_depth:
        return f"{type(msg).__name__}(...)"

    result = []
    prefix = indent * current_depth

    # Handle different message types
    if hasattr(msg, '__slots__'):
        for field in msg.__slots__:
            if field.startswith('_'):
                continue
            value = getattr(msg, field)

            # Format value based on type
            if isinstance(value, (int, float, str, bool)):
                result.append(f"{prefix}{field}: {value}")
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    result.append(f"{prefix}{field}: []")
                elif isinstance(value[0], (int, float)):
                    # Numeric array - show shape and range
                    arr = np.array(value)
                    result.append(f"{prefix}{field}: array{arr.shape} [{arr.min():.3f} to {arr.max():.3f}]")
                else:
                    result.append(f"{prefix}{field}: [{len(value)} items]")
                    if current_depth < max_depth - 1 and len(value) > 0:
                        result.append(format_message_sample(value[0], max_depth, current_depth + 1, indent))
            elif hasattr(value, '__slots__'):
                result.append(f"{prefix}{field}:")
                result.append(format_message_sample(value, max_depth, current_depth + 1, indent))
            else:
                result.append(f"{prefix}{field}: {type(value).__name__}")
    else:
        result.append(f"{prefix}{msg}")

    return "\n".join(result)


def analyze_message_content(msg):
    """Analyze message content and return useful statistics."""
    stats = {}

    if hasattr(msg, '__slots__'):
        for field in msg.__slots__:
            if field.startswith('_'):
                continue
            value = getattr(msg, field)

            if isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    arr = np.array(value)
                    stats[field] = {
                        'type': 'numeric_array',
                        'shape': arr.shape,
                        'min': float(arr.min()),
                        'max': float(arr.max()),
                        'mean': float(arr.mean())
                    }
                else:
                    stats[field] = {
                        'type': 'array',
                        'length': len(value)
                    }
            elif isinstance(value, (int, float)):
                stats[field] = {
                    'type': 'scalar',
                    'value': value
                }

    return stats


def check_bag_info(bag_path, verbose=False, show_samples=False):
    """
    Read and display information about a ROS bag file using AnyReader.

    Args:
        bag_path: Path to the .bag file
        verbose: If True, display detailed message information
        show_samples: If True, show sample messages from each topic
    """
    if not os.path.exists(bag_path):
        print(f"Error: File not found: {bag_path}")
        return

    print(f"\n{'='*100}")
    print(f"BAG FILE INFORMATION")
    print(f"{'='*100}")
    print(f"File: {bag_path}")
    print(f"Size: {format_bytes(os.path.getsize(bag_path))}")

    try:
        # Use AnyReader to read the bag file
        with AnyReader([Path(bag_path)]) as reader:
            # Collect topic information
            topic_info = defaultdict(lambda: {
                'count': 0,
                'msg_type': None,
                'first_msg': None,
                'last_msg': None,
                'first_time': None,
                'last_time': None
            })

            print(f"\n{'='*100}")
            print(f"Reading bag file...")
            print(f"{'='*100}")

            # Read all connections (topics)
            connections = list(reader.connections)
            print(f"\nFound {len(connections)} topic(s)")

            for conn in connections:
                print(f"  - {conn.topic} ({conn.msgtype})")

            # Count messages and collect samples
            print(f"\nCounting messages...")
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                msg_type = connection.msgtype

                topic_info[topic]['count'] += 1
                topic_info[topic]['msg_type'] = msg_type

                if topic_info[topic]['first_time'] is None:
                    topic_info[topic]['first_time'] = timestamp
                    if show_samples or verbose:
                        topic_info[topic]['first_msg'] = reader.deserialize(rawdata, msg_type)

                topic_info[topic]['last_time'] = timestamp
                if show_samples or verbose:
                    topic_info[topic]['last_msg'] = reader.deserialize(rawdata, msg_type)

            # Calculate duration
            if topic_info:
                all_start_times = [info['first_time'] for info in topic_info.values() if info['first_time']]
                all_end_times = [info['last_time'] for info in topic_info.values() if info['last_time']]

                if all_start_times and all_end_times:
                    start_time = min(all_start_times)
                    end_time = max(all_end_times)
                    duration_ns = end_time - start_time
                    duration_s = duration_ns / 1e9

                    print(f"\n{'='*100}")
                    print(f"TIME INFORMATION")
                    print(f"{'='*100}")
                    print(f"Duration: {format_duration(duration_ns)} ({duration_s:.2f} seconds)")
                    print(f"Start time: {start_time / 1e9:.6f} seconds")
                    print(f"End time: {end_time / 1e9:.6f} seconds")

            # Topic summary
            total_messages = sum(info['count'] for info in topic_info.values())

            print(f"\n{'='*100}")
            print(f"TOPIC SUMMARY")
            print(f"{'='*100}")
            print(f"Total topics: {len(topic_info)}")
            print(f"Total messages: {total_messages}")

            # Display topic details
            print(f"\n{'='*100}")
            print(f"TOPIC DETAILS")
            print(f"{'='*100}")
            print(f"{'Topic':<50} {'Type':<45} {'Count':<10} {'Freq (Hz)':<10}")
            print(f"{'-'*115}")

            # Sort topics by name
            for topic_name in sorted(topic_info.keys()):
                info = topic_info[topic_name]
                msg_type = info['msg_type']
                msg_count = info['count']

                # Calculate frequency
                if info['first_time'] and info['last_time']:
                    topic_duration = (info['last_time'] - info['first_time']) / 1e9
                    frequency = msg_count / topic_duration if topic_duration > 0 else 0
                else:
                    frequency = 0

                # Truncate long type names
                msg_type_short = msg_type if len(msg_type) <= 45 else msg_type[:42] + "..."
                topic_short = topic_name if len(topic_name) <= 50 else topic_name[:47] + "..."

                print(f"{topic_short:<50} {msg_type_short:<45} {msg_count:<10} {frequency:<10.2f}")

            # Group topics by message type
            print(f"\n{'='*100}")
            print(f"TOPICS GROUPED BY MESSAGE TYPE")
            print(f"{'='*100}")

            type_groups = defaultdict(list)
            for topic_name, info in topic_info.items():
                type_groups[info['msg_type']].append(topic_name)

            for msg_type in sorted(type_groups.keys()):
                print(f"\n{msg_type}:")
                for topic in sorted(type_groups[msg_type]):
                    info = topic_info[topic]
                    count = info['count']
                    if info['first_time'] and info['last_time']:
                        topic_duration = (info['last_time'] - info['first_time']) / 1e9
                        freq = count / topic_duration if topic_duration > 0 else 0
                    else:
                        freq = 0
                    print(f"  - {topic} ({count} msgs, {freq:.2f} Hz)")

            # Display sample messages if requested
            if show_samples and topic_info:
                print(f"\n{'='*100}")
                print(f"SAMPLE MESSAGES (first message of each topic)")
                print(f"{'='*100}")

                for topic in sorted(topic_info.keys()):
                    info = topic_info[topic]
                    if info['first_msg'] is not None:
                        print(f"\n{'-'*100}")
                        print(f"Topic: {topic}")
                        print(f"Type: {info['msg_type']}")
                        print(f"{'-'*100}")
                        print(format_message_sample(info['first_msg'], max_depth=3))

            # Display detailed analysis if verbose
            if verbose and topic_info:
                print(f"\n{'='*100}")
                print(f"DETAILED MESSAGE ANALYSIS")
                print(f"{'='*100}")

                for topic in sorted(topic_info.keys()):
                    info = topic_info[topic]
                    if info['first_msg'] is not None:
                        print(f"\n{'-'*100}")
                        print(f"Topic: {topic}")
                        print(f"{'-'*100}")

                        stats = analyze_message_content(info['first_msg'])
                        if stats:
                            for field, field_stats in stats.items():
                                if field_stats['type'] == 'numeric_array':
                                    print(f"  {field}:")
                                    print(f"    Shape: {field_stats['shape']}")
                                    print(f"    Range: [{field_stats['min']:.3f}, {field_stats['max']:.3f}]")
                                    print(f"    Mean: {field_stats['mean']:.3f}")
                                elif field_stats['type'] == 'array':
                                    print(f"  {field}: array with {field_stats['length']} items")
                                elif field_stats['type'] == 'scalar':
                                    print(f"  {field}: {field_stats['value']}")

            print(f"\n{'='*100}\n")

    except Exception as e:
        print(f"Error reading bag file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python check_bag_info.py <path_to_bag_file> [--verbose] [--samples]")
        print("\nOptions:")
        print("  --verbose, -v    Show detailed message analysis")
        print("  --samples, -s    Show sample messages from each topic")
        print("\nExample:")
        print('  python check_bag_info.py "D:\\DESKTOP\\push_block_000.bag"')
        print('  python check_bag_info.py "D:\\DESKTOP\\push_block_000.bag" --samples')
        print('  python check_bag_info.py "D:\\DESKTOP\\push_block_000.bag" --verbose --samples')
        sys.exit(1)

    bag_path = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    show_samples = '--samples' in sys.argv or '-s' in sys.argv

    check_bag_info(bag_path, verbose=verbose, show_samples=show_samples)


if __name__ == '__main__':
    main()