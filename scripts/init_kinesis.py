"""
init_kinesis.py — Create the Kinesis stream in LocalStack on first run.

Usage:
    python scripts/init_kinesis.py

Environment variables:
    AWS_ENDPOINT_URL        LocalStack endpoint (default: http://localhost:4566)
    KINESIS_STREAM_NAME     Stream name (default: flowwatch-telemetry)
    AWS_REGION              AWS region (default: us-east-1)
"""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError

ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
STREAM_NAME = os.getenv("KINESIS_STREAM_NAME", "flowwatch-telemetry")
REGION = os.getenv("AWS_REGION", "us-east-1")


def main() -> None:
    client = boto3.client(
        "kinesis",
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )

    # Check if stream already exists
    try:
        client.describe_stream_summary(StreamName=STREAM_NAME)
        print(f"Stream '{STREAM_NAME}' already exists — nothing to do.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            print(f"Error checking stream: {e}", file=sys.stderr)
            sys.exit(1)

    # Create stream
    client.create_stream(StreamName=STREAM_NAME, ShardCount=1)
    print(f"Stream '{STREAM_NAME}' created (1 shard) at {ENDPOINT_URL}")

    # Wait for it to become active
    waiter = client.get_waiter("stream_exists")
    waiter.wait(StreamName=STREAM_NAME)
    print(f"Stream '{STREAM_NAME}' is now ACTIVE.")


if __name__ == "__main__":
    main()
